"""
Generates a batch of run_evolution.py config JSON files by varying master_seed (and
optionally any other parameter) over a shared base template. Meant to be copied/edited
per batch -- change MASTER_SEEDS, FILENAME_PREFIX, and OVERRIDES below, then run.
"""

# ==== 1. IMPORTS ============================================================
import json
from pathlib import Path
import random 

# ==== 2. CONSTANTS / USER INPUTS ============================================
OUTPUT_FOLDER = Path("C:/EPANN_replay/configs/batch_to_run")
FILENAME_PREFIX = "ep_r250_"   # files are named f"{FILENAME_PREFIX}{seed}.json"

n_configs = 25   # number of configs to generate (and thus seeds to vary)

MASTER_SEEDS = [random.randint(0, 1000) for _ in range(n_configs)]
NOISE_SEEDS = [random.randint(0, 1000) for _ in range(n_configs)]
REWARD_SEEDS = [random.randint(0, 1000) for _ in range(n_configs)]
TEST_SEEDS = [random.randint(0, 1000) for _ in range(n_configs)]
WEIGHT_INIT_SEEDS = [random.randint(0, 1000) for _ in range(n_configs)]
# MASTER_SEEDS = [5, 2, 8, 34, 1]
# NOISE_SEEDS = [5, 2, 8, 34, 1]
# REWARD_SEEDS = [5, 2, 8, 34, 1]
# TEST_SEEDS = [5, 2, 8, 34, 1]
# WEIGHT_INIT_SEEDS = [5, 2, 8, 34, 1]
# MASTER_SEEDS = range(n_configs)   # 0..(n_configs-1)
# NOISE_SEEDS = range(n_configs)   # 0..(n_configs-1)
# REWARD_SEEDS = range(n_configs)   # 0..(n_configs-1)
# TEST_SEEDS = range(n_configs)   # 0..(n_configs-1)
# WEIGHT_INIT_SEEDS = range(n_configs)   # 0..(n_configs-1)


BASE_CONFIG = {
    "master_seed": None,   # overwritten per file by build_config()
    "device": "cuda",
    "n_neurons": 8,
    "noise_seed": None,
    "reward_seed": None,
    "test_seed": None,
    "weight_init_seed": None,
    "evo_context_cues_on": True,
    "evo_sensory_cues_on": True,
    "evo_plasticity_on": True,
    "paradigm": "trainA, 50, replay, 10, trainB, 50",
    "num_generations": 300,
    "search_popsize": 200,
    "radius_init": 250,
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
    "turn_reward_big": 1.0,
    "turn_reward_small": 0.0,
}

# applied on top of BASE_CONFIG for this batch, e.g. to switch off context cues:
OVERRIDES = {
    "evo_context_cues_on": True,
}


# ==== 3. CONFIG GENERATION FUNCTIONS ========================================
def build_config(master_seed, noise_seed, reward_seed, test_seed, weight_init_seed, base_config=BASE_CONFIG, overrides=OVERRIDES):
    """Merge base_config + overrides + this file's master_seed into one config dict."""
    return {**base_config, **overrides, "master_seed": master_seed, "noise_seed": noise_seed, "reward_seed": reward_seed, "test_seed": test_seed, "weight_init_seed": weight_init_seed}


def write_config_batch(master_seeds, noise_seeds, reward_seeds, test_seeds, weight_init_seeds, output_folder=OUTPUT_FOLDER, filename_prefix=FILENAME_PREFIX,
                        base_config=BASE_CONFIG, overrides=OVERRIDES):
    """Write one config JSON per seed in master_seeds into output_folder."""
    output_folder.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for seed in range(len(master_seeds)):
        config = build_config(seed, noise_seeds[seed], reward_seeds[seed], test_seeds[seed], weight_init_seeds[seed], base_config, overrides)
        path = output_folder / f"{filename_prefix}{seed}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        written_paths.append(path)
    return written_paths


# ==== 4. MAIN EXECUTION =====================================================
if __name__ == "__main__":
    paths = write_config_batch(MASTER_SEEDS, NOISE_SEEDS, REWARD_SEEDS, TEST_SEEDS, WEIGHT_INIT_SEEDS)
    print(f"Wrote {len(paths)} config file(s) to {OUTPUT_FOLDER}")
    for path in paths:
        print(f"  {path.name}")
