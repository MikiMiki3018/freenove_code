"""
Q-learning based gait optimization for Freenove Big Hexapod (FNK0052)

Integrated with your real code:

- IMU:        imu.IMU (update_imu_state -> pitch, roll, yaw)
- Ultrasonic: ultrasonic.Ultrasonic (get_distance -> cm)
- Control:    control.Control (run_gait -> real tripod/ripple motion)

Methodology alignment:
- State: (pitch_bin, terrain_bin)
- Action: discrete combinations of gait type (tripod/ripple) and speed
- Reward: estimated forward distance - penalties for instability/collisions
- Policy: epsilon-greedy
- Update rule: tabular Q-learning
- Baseline comparison: learned gait D/T vs fixed tripod D/T
- Stability threshold: >= 30 seconds stable time per episode
- Convergence: stop when Q-table change norm < tolerance

Logging:
- step_log.csv      : per-step IMU, servo angles, distance, reward, stability
- episode_log.csv   : per-episode reward, speed, stable time, Q-change
- baseline_log.csv  : baseline tripod gait performance
"""

import time
import csv
from datetime import datetime
import numpy as np
import random

from control import Control
from ultrasonic import Ultrasonic

###########################################################
#  CONSTANTS & CONFIGURATION
###########################################################

# Discrete state space sizes
N_PITCH_BIN = 3      # low / normal / high tilt
N_TERRAIN_BIN = 2    # flat / obstacle-near

# Actions:
# 0 = tripod, slow
# 1 = tripod, medium
# 2 = tripod, fast
# 3 = ripple, medium
# 4 = ripple, fast
N_ACTIONS = 5

# Q-learning hyperparameters
ALPHA = 0.2        # learning rate
GAMMA = 0.9        # discount factor
EPSILON_START = 0.3
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

# Training config
EPISODES = 50           # adjust depending on time available
STEPS_PER_EPISODE = 10  # Q-steps per episode
CONVERGENCE_TOL = 1e-3  # threshold for Q-table change (convergence)

# Reward shaping
INSTABILITY_PENALTY = 30.0
COLLISION_PENALTY = 50.0   # if ultrasonic detects very near obstacle

# Stability threshold (methodology)
STABILITY_TIME_THRESHOLD = 30.0  # seconds per episode

# Baseline improvement requirement (>= 10%)
BASELINE_IMPROVEMENT_TARGET = 1.10

# Logging file paths
STEP_LOG_FILE = "step_log.csv"
EPISODE_LOG_FILE = "episode_log.csv"
BASELINE_LOG_FILE = "baseline_log.csv"


###########################################################
#  LOGGING INITIALIZATION
###########################################################

def init_logs():
    """Create CSV headers for step-level and episode-level logs."""
    with open(STEP_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "episode",
            "step",
            "servo_x",
            "servo_y",
            "servo_angles",
            "pitch",
            "roll",
            "yaw",
            "cycle_time",
            "distance",
            "reward",
            "stable_flag"
        ])

    with open(EPISODE_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "episode",
            "total_reward",
            "avg_reward",
            "total_distance",
            "avg_speed",
            "stable_time",
            "meets_stability_threshold",
            "q_change_norm"
        ])

    with open(BASELINE_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "trial",
            "total_distance",
            "total_time",
            "distance_per_time"
        ])


def log_step(episode, step, servo_x, servo_y, servo_angles,
             pitch, roll, yaw, cycle_time, distance,
             reward, stable_flag):
    """Log one Q-learning step."""
    with open(STEP_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(),
            episode,
            step,
            servo_x,
            servo_y,
            servo_angles,
            pitch,
            roll,
            yaw,
            cycle_time,
            distance,
            reward,
            int(stable_flag)
        ])


def log_episode(episode, total_reward, avg_reward, total_distance,
                avg_speed, stable_time, meets_stability_threshold,
                q_change_norm):
    """Log one Q-learning episode summary."""
    with open(EPISODE_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(),
            episode,
            total_reward,
            avg_reward,
            total_distance,
            avg_speed,
            stable_time,
            int(meets_stability_threshold),
            q_change_norm
        ])


def log_baseline_trial(trial_idx, total_distance, total_time, dpt):
    """Log a baseline fixed-gait trial."""
    with open(BASELINE_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(),
            trial_idx,
            total_distance,
            total_time,
            dpt
        ])


###########################################################
#  HEXAPOD ENVIRONMENT (STATE / ACTION / REWARD)
###########################################################

class HexapodEnv:
    """
    Environment wrapper for the Freenove hexapod using your real control code.

    Responsibilities:
    - Define state: (pitch_bin, terrain_bin)
    - Execute actions via Control.run_gait:
        actions choose gait type (tripod/ripple) and speed
    - Estimate reward:
        estimated forward distance - instability/collision penalties
    - Provide logging info:
        servo position (approx X-Y), servo joint angles, IMU, stability flag
    """

    def __init__(self):
        self.control = Control()        # your real control class
        self.imu = self.control.imu     # reuse IMU instance created inside Control
        self.ultra = Ultrasonic()       # your ultrasonic class

        # For distance estimation
        self.total_distance = 0.0  # cumulative forward distance (cm)
        self.base_step_distance_cm = 5.0  # nominal per-step distance (scaled by speed)

        # For reward
        self.last_speed_param = 6  # default mid-speed

    # ===== Sensor access using your real code =====

    def read_imu(self):
        """
        Read IMU values from your IMU class.

        IMU.update_imu_state() returns: (pitch, roll, yaw) in degrees.
        """
        pitch, roll, yaw = self.imu.update_imu_state()
        return pitch, roll, yaw

    def read_ultrasonic_distance(self):
        """
        Read ultrasonic distance (cm) using your Ultrasonic class.
        If None or error, treat as "no obstacle".
        """
        dist = self.ultra.get_distance()
        if dist is None:
            return 300.0  # effectively "no obstacle nearby"
        return dist

    def get_servo_angles(self):
        """
        Return current servo joint angles (6x3) from Control as a flat list.
        """
        angles = self.control.current_angles  # 6 x 3 list
        flat = []
        for leg in angles:
            flat.extend(leg)
        return flat

    def estimate_position(self):
        """
        Approximate robot position in 2D.

        We don't have odometry, so we just use:
            x = total_distance (forward direction)
            y = 0
        """
        return self.total_distance, 0.0

    # ===== State, reward, and step =====

    def discretize_pitch(self, pitch):
        """
        Map continuous pitch to 3 bins: low / normal / high.
        """
        if pitch < -3:
            return 0   # nose-down
        elif pitch < 3:
            return 1   # relatively level
        else:
            return 2   # nose-up / unstable

    def discretize_terrain(self, dist_cm):
        """
        Discretize terrain based on ultrasonic distance:
        0: flat / no nearby obstacle
        1: obstacle nearby (closer than 30 cm)
        """
        return 1 if dist_cm < 30.0 else 0

    def is_unstable(self, pitch, roll):
        """
        Determine if robot is unstable using IMU.
        """
        return abs(pitch) > 9.0 or abs(roll) > 9.0

    def get_state(self):
        """
        Construct the discrete state tuple: (pitch_bin, terrain_bin)
        """
        pitch, roll, yaw = self.read_imu()
        pitch_bin = self.discretize_pitch(pitch)
        dist_cm = self.read_ultrasonic_distance()
        terrain_bin = self.discretize_terrain(dist_cm)
        return (pitch_bin, terrain_bin), (pitch, roll, yaw, dist_cm)

    def execute_gait_action(self, action):
        """
        Map discrete action index to your real Control.run_gait() function.

        Actions:
        0: tripod, slow    (gait='1', speed_param=4)
        1: tripod, medium  (gait='1', speed_param=6)
        2: tripod, fast    (gait='1', speed_param=8)
        3: ripple, medium  (gait='2', speed_param=6)
        4: ripple, fast    (gait='2', speed_param=8)

        We keep forward motion along +Y direction, angle=0.
        """
        if action == 0:
            gait_type = "1"      # tripod
            speed_param = 4
        elif action == 1:
            gait_type = "1"      # tripod
            speed_param = 6
        elif action == 2:
            gait_type = "1"      # tripod
            speed_param = 8
        elif action == 3:
            gait_type = "2"      # ripple
            speed_param = 6
        else:
            gait_type = "2"      # ripple
            speed_param = 8

        self.last_speed_param = speed_param

        # data format as in your run_gait:
        # data = ['CMD_MOVE', gait, x, y, speed_param, angle]
        x = 0      # no lateral
        y = 25     # forward command
        angle = 0  # no rotation

        data = ['CMD_MOVE',
                gait_type,
                str(x),
                str(y),
                str(speed_param),
                str(angle)]

        # Z is leg lift height; F is initial cycle count, but your run_gait
        # recomputes effective F based on speed_param.
        self.control.run_gait(data, Z=40, F=64)

    def estimate_step_distance(self):
        """
        Estimate forward distance moved in the last step (cm).

        We don't have odometry, so we use a simple model:
            distance ∝ base_step_distance * (speed_param / 10)
        """
        return self.base_step_distance_cm * (self.last_speed_param / 10.0)

    def reset(self):
        """
        Reset environment for a new episode:

        - Reset internal distance counter
        - Move robot back to neutral position
        - Read initial state
        """
        self.total_distance = 0.0
        self.control.move_position(0, 0, 0)
        time.sleep(1.0)
        state, _ = self.get_state()
        return state

    def step(self, action, episode, step_idx):
        """
        Execute one Q-learning step:
        - Run a real gait via Control.run_gait()
        - Measure IMU, ultrasonic, time
        - Estimate distance
        - Compute reward and next state
        - Return transition and logging info
        """
        t0 = time.time()

        # Execute physical motion
        self.execute_gait_action(action)

        # Measure IMU after moving
        pitch, roll, yaw = self.read_imu()

        # Measure terrain / collision
        dist_cm = self.read_ultrasonic_distance()
        collision = dist_cm < 10.0

        # Estimate distance moved
        distance = self.estimate_step_distance()
        self.total_distance += distance

        # Measure cycle time
        cycle_time = time.time() - t0

        # Check stability
        unstable = self.is_unstable(pitch, roll)
        stable_flag = not unstable

        # Compute reward
        reward = distance
        if unstable:
            reward -= INSTABILITY_PENALTY
        if collision:
            reward -= COLLISION_PENALTY

        # Next state
        next_state, _ = self.get_state()

        # Logging-related data
        servo_angles = self.get_servo_angles()
        servo_x, servo_y = self.estimate_position()

        info = {
            "pitch": pitch,
            "roll": roll,
            "yaw": yaw,
            "cycle_time": cycle_time,
            "distance": distance,
            "servo_x": servo_x,
            "servo_y": servo_y,
            "servo_angles": servo_angles,
            "stable_flag": stable_flag
        }

        return next_state, reward, info


###########################################################
#  Q-LEARNING AGENT
###########################################################

class QLearningAgent:
    """
    Tabular Q-learning agent on discrete state space (pitch_bin, terrain_bin).

    - Q-table shape: [pitch_bin][terrain_bin][actions]
    - Epsilon-greedy exploration
    - Q-change norm for convergence check
    """

    def __init__(self):
        self.q = np.zeros((N_PITCH_BIN, N_TERRAIN_BIN, N_ACTIONS))
        self.epsilon = EPSILON_START

    def choose_action(self, state):
        p, t = state
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        else:
            return int(np.argmax(self.q[p, t, :]))

    def update(self, state, action, reward, next_state):
        p, t = state
        np_, nt = next_state

        current_value = self.q[p, t, action]
        next_max = np.max(self.q[np_, nt, :])

        td_target = reward + GAMMA * next_max
        td_error = td_target - current_value
        self.q[p, t, action] += ALPHA * td_error

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def q_change_norm(self, old_q):
        """
        L2 norm of Q-table change, used for convergence check.
        """
        diff = self.q - old_q
        return np.linalg.norm(diff)


###########################################################
#  BASELINE GAIT (FIXED TRIPOD) COMPARISON
###########################################################

def run_baseline(env, trials=3, steps_per_trial=10, baseline_action=1):
    """
    Run a fixed gait (tripod, medium speed) without learning.

    Uses env.step(baseline_action, ...) repeatedly and measures:
        distance/time ratio as baseline.

    Returns:
        baseline_distance_per_time (float)
    """
    print("\n=== Running Baseline Gait (Tripod, medium speed) ===")
    total_ratios = []

    for t in range(trials):
        print(f"\n[Baseline Trial {t+1}/{trials}]")
        state = env.reset()
        total_distance = 0.0
        t_start = time.time()

        for step_idx in range(steps_per_trial):
            next_state, reward, info = env.step(baseline_action, -1, step_idx)
            total_distance += info["distance"]

        total_time = time.time() - t_start
        if total_time > 0:
            dpt = total_distance / total_time
        else:
            dpt = 0.0

        print(f"[Baseline] Distance={total_distance:.2f} cm, Time={total_time:.2f} s, D/T={dpt:.4f}")
        log_baseline_trial(t, total_distance, total_time, dpt)
        total_ratios.append(dpt)

    baseline = np.mean(total_ratios) if total_ratios else 0.0
    print(f"\n[Baseline] Average distance/time ratio = {baseline:.4f}")
    return baseline


###########################################################
#  TRAINING LOOP WITH CONVERGENCE & STABILITY CHECKS
###########################################################

def train_q_learning():
    """
    Main Q-learning training loop.

    - Initializes logs
    - Runs baseline tripod gait to establish comparison metric
    - Runs Q-learning episodes with:
        * step-level logging
        * episode-level reward, speed, stability time
        * Q-table change norm for convergence
    - Checks convergence and stops early if Q stabilizes
    - At the end compares learned gait performance vs baseline
    """
    init_logs()
    env = HexapodEnv()
    agent = QLearningAgent()

    # 1) Baseline evaluation (fixed tripod, medium speed, action=1)
    baseline_dpt = run_baseline(env, trials=3, steps_per_trial=STEPS_PER_EPISODE, baseline_action=1)

    # 2) Q-learning episodes
    print("\n=== Starting Q-learning Training ===")
    last_episode_avg_speed = 0.0  # keep track for final comparison

    for ep in range(EPISODES):
        print(f"\n========== EPISODE {ep+1}/{EPISODES} ==========")
        state = env.reset()

        total_reward = 0.0
        total_distance = 0.0
        total_cycle_time = 0.0
        stable_time = 0.0

        old_q = np.copy(agent.q)

        for step_idx in range(STEPS_PER_EPISODE):
            print(f"\n--- Step {step_idx+1}/{STEPS_PER_EPISODE} ---")

            # 1. Choose action (epsilon-greedy)
            action = agent.choose_action(state)
            print(f"[EP {ep+1}] Chosen action: {action}")

            # 2. Take step in environment
            next_state, reward, info = env.step(action, ep, step_idx)

            # 3. Q-learning update
            agent.update(state, action, reward, next_state)

            # 4. Accumulate stats
            total_reward += reward
            total_distance += info["distance"]
            total_cycle_time += info["cycle_time"]
            if info["stable_flag"]:
                stable_time += info["cycle_time"]

            # 5. Log step
            log_step(
                episode=ep,
                step=step_idx,
                servo_x=info["servo_x"],
                servo_y=info["servo_y"],
                servo_angles=info["servo_angles"],
                pitch=info["pitch"],
                roll=info["roll"],
                yaw=info["yaw"],
                cycle_time=info["cycle_time"],
                distance=info["distance"],
                reward=reward,
                stable_flag=info["stable_flag"]
            )

            # Move to next state
            state = next_state

        # Episode summary
        avg_reward = total_reward / STEPS_PER_EPISODE
        if total_cycle_time > 0:
            avg_speed = total_distance / total_cycle_time
        else:
            avg_speed = 0.0

        q_change = agent.q_change_norm(old_q)
        meets_stability = stable_time >= STABILITY_TIME_THRESHOLD

        last_episode_avg_speed = avg_speed  # save for final comparison

        # Log episode
        log_episode(
            episode=ep,
            total_reward=total_reward,
            avg_reward=avg_reward,
            total_distance=total_distance,
            avg_speed=avg_speed,
            stable_time=stable_time,
            meets_stability_threshold=meets_stability,
            q_change_norm=q_change
        )

        print(f"[EP {ep+1} Summary] Reward={total_reward:.2f}, Distance={total_distance:.2f} cm, "
              f"AvgSpeed={avg_speed:.4f} cm/s, StableTime={stable_time:.2f} s, "
              f"Q-change={q_change:.6f}, StabilityOK={meets_stability}")

        # 3) Convergence check (Q-values stabilize)
        if q_change < CONVERGENCE_TOL:
            print(f"\n[Convergence] Q-table change {q_change:.6f} < {CONVERGENCE_TOL}. Stopping early.")
            break

        # 4) Decay exploration
        agent.decay_epsilon()

    # 5) Final evaluation vs baseline
    print("\n=== Final Evaluation vs Baseline ===")
    learned_dpt = last_episode_avg_speed
    if baseline_dpt > 0:
        improvement = learned_dpt / baseline_dpt
    else:
        improvement = 0.0

    print(f"Baseline D/T: {baseline_dpt:.4f}, Learned D/T: {learned_dpt:.4f}, "
          f"Improvement factor: {improvement:.3f}")

    if improvement >= BASELINE_IMPROVEMENT_TARGET:
        print("[Goal] Learned gait outperforms baseline by >= 10%.")
    else:
        print("[Goal] Improvement < 10%. You can increase EPISODES or tune reward shaping.")


###########################################################
#  MAIN ENTRY POINT
###########################################################

if __name__ == "__main__":
    train_q_learning()
