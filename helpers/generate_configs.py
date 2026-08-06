"""
Generates a batch of run_evolution.py config JSON files by varying master_seed (and
optionally any other parameter) over a shared base template. Meant to be copied/edited
per batch -- change MASTER_SEEDS, FILENAME_PREFIX, and OVERRIDES below, then run.
"""

# ==== 1. IMPORTS ============================================================
import json
from pathlib import Path

# ==== 2. CONSTANTS / USER INPUTS ============================================
OUTPUT_FOLDER = Path("C:/EPANN_replay/configs/batch_to_run")
FILENAME_PREFIX = "ep_soltoggio_rewards_no_cc_"   # files are named f"{FILENAME_PREFIX}{seed}.json"

MASTER_SEEDS = range(25)   # 0..24

BASE_CONFIG = {
    "master_seed": None,   # overwritten per file by build_config()
    "device": "cuda",
    "n_neurons": 15,
    "noise_seed": 1,
    "reward_seed": 2,
    "test_seed": 3,
    "weight_init_seed": 4,
    "evo_context_cues_on": True,
    "evo_sensory_cues_on": True,
    "evo_plasticity_on": True,
    "paradigm": "trainA, 50, replay, 10, trainB, 50",
    "num_generations": 500,
    "search_popsize": 150,
    "radius_init": 50,
    "stdev_learning_rate": 0.1,
    "momentum": 0.9,
    "l1_lambda": 0.001,
    "tracked_per_interval": 50,
    "max_networks_preview": 6,
    "max_runs_preview": 20,
    "hist_bin_width": 1,
    "dt": 0.2,
    "tau": 1.0,
    "noise_std": 0.1,
    "straight_thresh": 1 / 3,
    "big_reward": 1.0,
    "small_reward": 0.2,
    "crash_penalty": -0.4,
    "turn_reward_big": 0.0,
    "turn_reward_small": 0.0,
}

# applied on top of BASE_CONFIG for this batch, e.g. to switch off context cues:
OVERRIDES = {
    "evo_context_cues_on": False,
}


# ==== 3. CONFIG GENERATION FUNCTIONS ========================================
def build_config(master_seed, base_config=BASE_CONFIG, overrides=OVERRIDES):
    """Merge base_config + overrides + this file's master_seed into one config dict."""
    return {**base_config, **overrides, "master_seed": master_seed}


def write_config_batch(master_seeds, output_folder=OUTPUT_FOLDER, filename_prefix=FILENAME_PREFIX,
                        base_config=BASE_CONFIG, overrides=OVERRIDES):
    """Write one config JSON per seed in master_seeds into output_folder."""
    output_folder.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for seed in master_seeds:
        config = build_config(seed, base_config, overrides)
        path = output_folder / f"{filename_prefix}{seed}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        written_paths.append(path)
    return written_paths


# ==== 4. MAIN EXECUTION =====================================================
if __name__ == "__main__":
    paths = write_config_batch(MASTER_SEEDS)
    print(f"Wrote {len(paths)} config file(s) to {OUTPUT_FOLDER}")
    for path in paths:
        print(f"  {path.name}")
