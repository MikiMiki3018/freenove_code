import os
import shutil

def detect_next_trial():
    """Detect the next unused trial number."""
    trial_num = 1
    while os.path.exists(f"trial{trial_num}"):
        trial_num += 1
    return trial_num

def save_logs():
    # Files generated from each Q-learning run
    logs = {
        "step_log.csv": "step_log.csv",
        "episode_log.csv": "episode_log.csv",
        "baseline_log.csv": "baseline_log.csv"
    }

    trial_num = detect_next_trial()
    trial_folder = f"trial{trial_num}"

    print(f"\n=== Saving logs into folder: {trial_folder}/ ===")

    # Create folder
    os.makedirs(trial_folder, exist_ok=True)

    for original_name, file_name in logs.items():
        if os.path.exists(file_name):
            new_name = f"{trial_folder}/trial{trial_num}_{original_name}"
            shutil.move(file_name, new_name)
            print(f"Saved: {new_name}")
        else:
            print(f"WARNING: {file_name} not found. Did you run Q-learning first?")

    print("\nLogs saved successfully.\n")

if __name__ == "__main__":
    save_logs()
