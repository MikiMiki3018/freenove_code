"""
Q-learning based gait optimization for Freenove Big Hexapod (FNK0052)
(MODIFIED with: EPISODES=5, forward speed=10, and ENTER prompt before episodes)
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
N_PITCH_BIN = 3
N_TERRAIN_BIN = 2

# Actions
# We will reduce forward speed later in execute_gait_action()
N_ACTIONS = 5

# Q-learning hyperparameters
ALPHA = 0.2
GAMMA = 0.9
EPSILON_START = 0.3
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

# 5 EPISODES
EPISODES = 5
STEPS_PER_EPISODE = 10
CONVERGENCE_TOL = 1e-3

# Rewards
INSTABILITY_PENALTY = 30.0
COLLISION_PENALTY = 50.0

STABILITY_TIME_THRESHOLD = 30.0
BASELINE_IMPROVEMENT_TARGET = 1.10

# Logging
STEP_LOG_FILE = "step_log.csv"
EPISODE_LOG_FILE = "episode_log.csv"
BASELINE_LOG_FILE = "baseline_log.csv"

###########################################################
#  LOGGING FUNCTIONS
###########################################################

def init_logs():
    with open(STEP_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp","episode","step",
            "servo_x","servo_y","servo_angles",
            "pitch","roll","yaw",
            "cycle_time","distance","reward","stable_flag"
        ])

    with open(EPISODE_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp","episode","total_reward","avg_reward",
            "total_distance","avg_speed","stable_time",
            "meets_stability_threshold","q_change_norm"
        ])

    with open(BASELINE_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp","trial","total_distance","total_time","distance_per_time"
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
#  ENVIRONMENT
###########################################################

class HexapodEnv:
    def __init__(self):
        self.control = Control()
        self.imu = self.control.imu
        self.ultra = Ultrasonic()
        self.total_distance = 0.0
        self.base_step_distance_cm = 5.0
        self.last_speed_param = 6

    def read_imu(self):
        pitch, roll, yaw = self.imu.update_imu_state()
        return pitch, roll, yaw

    def read_ultrasonic_distance(self):
        d = self.ultra.get_distance()
        return 300.0 if d is None else d

    def get_servo_angles(self):
        flat = []
        for leg in self.control.current_angles:
            flat.extend(leg)
        return flat

    def estimate_position(self):
        return self.total_distance, 0.0

    def discretize_pitch(self, pitch):
        if pitch < -3: return 0
        elif pitch < 3: return 1
        return 2

    def discretize_terrain(self, dist):
        return 1 if dist < 30.0 else 0

    def is_unstable(self, pitch, roll):
        return abs(pitch) > 9 or abs(roll) > 9

    def get_state(self):
        pitch, roll, yaw = self.read_imu()
        dist = self.read_ultrasonic_distance()
        return (self.discretize_pitch(pitch), self.discretize_terrain(dist)), \
               (pitch, roll, yaw, dist)

    def execute_gait_action(self, action):
        """
        ⭐ MODIFIED: reduced forward speed y from 25 → 10 
        """
        if action == 0:
            gait_type, speed = "1", 4
        elif action == 1:
            gait_type, speed = "1", 6
        elif action == 2:
            gait_type, speed = "1", 8
        elif action == 3:
            gait_type, speed = "2", 6
        else:
            gait_type, speed = "2", 8

        self.last_speed_param = speed

        x = 0
        y = 10      # LOWERED FORWARD SPEED
        angle = 0

        data = ["CMD_MOVE", gait_type, str(x), str(y), str(speed), str(angle)]
        self.control.run_gait(data, Z=40, F=64)

    def estimate_step_distance(self):
        return self.base_step_distance_cm * (self.last_speed_param / 10.0)

    def reset(self):
        self.total_distance = 0
        self.control.move_position(0,0,0)
        time.sleep(1)
        s,_ = self.get_state()
        return s

    def step(self, action, episode, step_idx):
        t0 = time.time()
        self.execute_gait_action(action)

        pitch, roll, yaw = self.read_imu()
        dist_cm = self.read_ultrasonic_distance()
        collision = dist_cm < 10

        distance = self.estimate_step_distance()
        self.total_distance += distance

        cycle_time = time.time() - t0
        unstable = self.is_unstable(pitch, roll)

        reward = distance
        if unstable: reward -= INSTABILITY_PENALTY
        if collision: reward -= COLLISION_PENALTY

        next_state, _ = self.get_state()
        servo_angles = self.get_servo_angles()
        servo_x, servo_y = self.estimate_position()

        info = {
            "pitch": pitch, "roll": roll, "yaw": yaw,
            "cycle_time": cycle_time, "distance": distance,
            "servo_x": servo_x, "servo_y": servo_y,
            "servo_angles": servo_angles,
            "stable_flag": not unstable
        }

        return next_state, reward, info

###########################################################
#  Q-LEARNING AGENT
###########################################################

class QLearningAgent:
    def __init__(self):
        self.q = np.zeros((N_PITCH_BIN, N_TERRAIN_BIN, N_ACTIONS))
        self.epsilon = EPSILON_START

    def choose_action(self, state):
        p,t = state
        if random.random() < self.epsilon:
            return random.randint(0,N_ACTIONS-1)
        return int(np.argmax(self.q[p,t,:]))

    def update(self, state, action, reward, next_state):
        p,t = state
        pn, tn = next_state
        current = self.q[p,t,action]
        target = reward + GAMMA * np.max(self.q[pn,tn,:])
        self.q[p,t,action] += ALPHA*(target-current)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)

    def q_change_norm(self, old_q):
        return np.linalg.norm(self.q - old_q)

###########################################################
#  BASELINE
###########################################################

def run_baseline(env, trials=3, steps_per_trial=10, baseline_action=1):
    print("\n=== Running Baseline (Tripod, medium speed) ===")
    ratios = []

    for t in range(trials):
        input(f"\nPlace robot for baseline trial {t+1} and press ENTER...")
        state = env.reset()

        dist_total = 0
        t0 = time.time()

        for s in range(steps_per_trial):
            _,_,info = env.step(baseline_action, -1, s)
            dist_total += info["distance"]

        t_total = time.time() - t0
        dpt = dist_total/t_total if t_total>0 else 0

        ratios.append(dpt)
        log_baseline_trial(t, dist_total, t_total, dpt)

        print(f"Trial {t+1}: Dist={dist_total:.2f}cm  Time={t_total:.2f}s  D/T={dpt:.4f}")

    return np.mean(ratios)

###########################################################
#  TRAINING LOOP
###########################################################

def train_q_learning():
    init_logs()
    env = HexapodEnv()
    agent = QLearningAgent()

    baseline_dpt = run_baseline(env, steps_per_trial=STEPS_PER_EPISODE)

    print("\n=== Starting Q-learning (5 Episodes) ===")
    last_avg_speed = 0

    for ep in range(EPISODES):

        # Added manual pause so you can reposition robot
        input(f"\nPlace robot safely for EPISODE {ep+1} then press ENTER...")

        print(f"\n========== EPISODE {ep+1} ==========")
        state = env.reset()

        total_reward = 0
        total_distance = 0
        total_cycle_time = 0
        stable_time = 0

        old_q = np.copy(agent.q)

        for step_i in range(STEPS_PER_EPISODE):
            print(f"\n--- Step {step_i+1}/{STEPS_PER_EPISODE} ---")

            action = agent.choose_action(state)
            print(f"Action chosen: {action}")

            next_state, reward, info = env.step(action, ep, step_i)

            agent.update(state, action, reward,next_state)

            total_reward += reward
            total_distance += info["distance"]
            total_cycle_time += info["cycle_time"]
            if info["stable_flag"]:
                stable_time += info["cycle_time"]

            log_step(ep, step_i, info["servo_x"], info["servo_y"], info["servo_angles"],
                     info["pitch"], info["roll"], info["yaw"], info["cycle_time"],
                     info["distance"], reward, info["stable_flag"])

            state = next_state

        avg_reward = total_reward/STEPS_PER_EPISODE
        avg_speed = total_distance/total_cycle_time if total_cycle_time>0 else 0

        last_avg_speed = avg_speed

        q_change = agent.q_change_norm(old_q)
        stable_ok = stable_time >= STABILITY_TIME_THRESHOLD

        log_episode(ep, total_reward, avg_reward, total_distance,
                    avg_speed, stable_time, stable_ok, q_change)

        print(f"[EP {ep+1}] TotalReward={total_reward:.2f}, Dist={total_distance:.2f}, "
              f"AvgSpeed={avg_speed:.3f}, StabilityOK={stable_ok}")

        if q_change < CONVERGENCE_TOL:
            print("\nConverged. Stopping early.")
            break

        agent.decay_epsilon()

    print("\n=== Final Comparison ===")
    improvement = last_avg_speed/baseline_dpt if baseline_dpt>0 else 0
    print(f"Baseline D/T={baseline_dpt:.4f}, Learned D/T={last_avg_speed:.4f}, "
          f"Improvement={improvement:.3f}")

###########################################################
#  RUN
###########################################################

if __name__ == "__main__":
    train_q_learning()
