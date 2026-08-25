"""
Generates a batch of run_evolution.py config JSON files that sweep over one or more
parameters (the full cartesian product of SWEEP_PARAMS), with N_CONFIGS_PER_GROUP
seeded configs per parameter combination. Meant to be copied/edited per sweep --
change SWEEP_PARAMS, N_CONFIGS_PER_GROUP, SEED_MODE, FILENAME_PREFIX, and
BASE_CONFIG/OVERRIDES below, then run.

Example: SWEEP_PARAMS = {"n_neurons": [9, 20, 60], "evo_context_cues_on": [True, False],
"search_popsize": [200, 400, 40]} with N_CONFIGS_PER_GROUP = 20 generates
3 * 2 * 3 = 18 parameter combinations x 20 configs each = 360 config files, identical
in every field except the swept parameters (fixed within a combination) and the
seeds (vary within each combination, per SEED_MODE below).
"""

# ==== 1. IMPORTS ============================================================
import itertools
import json
from pathlib import Path
import random

# ==== 2. CONSTANTS / USER INPUTS ============================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = PROJECT_ROOT / "configs" / "generated_sweep"
FILENAME_PREFIX = "test _"   # files are named f"{FILENAME_PREFIX}{combo_slug}_{i}.json"

# Parameters to sweep -- the cartesian product of every list below becomes one
# parameter combination (one "group" of N_CONFIGS_PER_GROUP configs). Keys must
# match BASE_CONFIG keys; add/remove sweep params freely, this file doesn't need
# any other changes to keep working.
SWEEP_PARAMS = {
    "radius_init": [400],
    "search_popsize": [10000],
    "stdev_learning_rate": [0.1],
}

N_CONFIGS_PER_GROUP = 200   # number of seeded configs generated per parameter combination

# ---- seeds: same 5 seed types as generate_configs.py. One list per type is
# generated per SEED_MODE below, each of length
# len(sweep combinations) * N_CONFIGS_PER_GROUP -- unless SAME_RANDOMS_PER_GROUP
# is True, in which case only N_CONFIGS_PER_GROUP seeds are generated and that
# same list is reused for every combination (so e.g. config #3 of every parameter
# combination shares its seeds, controlling for RNG when comparing across combos). ----
SEED_MODE = "random"   # "random" | "incrementing" | "fixed"
SAME_RANDOMS_PER_GROUP = True   # True = reuse one seed list across every combination
# only read when SEED_MODE == "fixed" -- each list must have exactly
# N_CONFIGS_PER_GROUP entries if SAME_RANDOMS_PER_GROUP else
# len(sweep combinations) * N_CONFIGS_PER_GROUP entries (checked at generation time)
FIXED_SEEDS = {
    "master_seed": [],
    "noise_seed": [],
    "reward_seed": [],
    "test_seed": [],
    "weight_init_seed": [],
}


# Every other parameter -- same value in every generated config regardless of
# SWEEP_PARAMS/seeds above. Edit freely; the "master_seed"/etc. entries here are
# always overwritten per config (see build_config()).
BASE_CONFIG = {
    # ---- seeds ----
    "master_seed": None,   # overwritten per file by build_config()
    "noise_seed": None,
    "reward_seed": None,
    "test_seed": None,
    "weight_init_seed": None,

    # ---- evolutionary search: method choice + the hyperparameters that concern it.
    # Exactly ONE es_method-specific block below should be uncommented at a time -- see
    # runners/run_evolution.py's `match ES_METHOD` gating, which reads exactly the keys
    # for whichever es_method is active and nothing else. Comment/uncomment the whole
    # block (and flip es_method below to match) when switching which algorithm this
    # batch generates configs for -- don't delete the inactive block, it's the reference
    # copy for switching back.
    #
    # search_popsize/num_generations are read the same way by every es_method, but their
    # *sensible* value differs a lot by algorithm (PGPE samples a large population fresh
    # from one shared search distribution every generation; Cosyne maintains a much
    # smaller population of literally-persisting individuals) -- kept as single
    # non-duplicated keys here, so remember to also revisit their value when switching. ----
    "es_method": "cosyne",
    "num_generations": 300,
    "search_popsize": 50,  # Cosyne's own documented example (see cosyne_* below) uses 50 --
                            # PGPE ran this same key at 200, bump it back up when re-enabling PGPE

    # ---- PGPE hyperparameters (see runners/run_evolution.py). Currently commented out --
    # es_method above is "cosyne". Anti-stagnation measures are all independently opt-in via
    # their own "*_enabled" flag; off here, matching run_evolution.py's own CONFIG.get(...)
    # None-if-absent behavior. Their value params stay listed (with placeholder values) even
    # while disabled, so flipping an "*_enabled" flag back on never trips run_evolution.py's
    # fail-loud missing-key check. ----
    # "radius_init": 200,
    # "stdev_learning_rate": 0.01,
    # "momentum": 0.9,
    # "pgpe_stdev_min_enabled": False,
    # "pgpe_stdev_min": 0.01,
    # "pgpe_restart_enabled": False,
    # "pgpe_restart_patience": 50,
    # "pgpe_restart_min_improvement": 0.01,
    # "pgpe_restart_radius": 200,
    # "pgpe_center_perturb_enabled": False,
    # "pgpe_center_perturb_interval": 50,
    # "pgpe_center_perturb_std": 0.1,
    # "pgpe_perturb_seed": 12345,

    # ---- Cosyne hyperparameters (see runners/run_evolution.py). Active -- es_method above
    # is "cosyne". Values are EvoTorch's own documented example (docs.evotorch.ai's "Gym
    # Experiments with PGPE and CoSyNE" notebook, its Cosyne(...) call), not independently
    # tuned yet -- a real tuning pass is still to come. cosyne_elitism_ratio/cosyne_eta/
    # cosyne_num_children are explicitly None here, not omitted (this project never lets code
    # silently fall back to a default -- see epann-code-style-principles): that notebook's
    # Cosyne(...) call leaves them unset, which evotorch's own Cosyne.__init__ treats as
    # "elitism via num_elites instead of elitism_ratio" / "one-point cross-over instead of
    # SBX" / "num_children = popsize/2" -- these are this run's genuinely CHOSEN values
    # (None), written down instead of left implicit. num_elites and elitism_ratio are
    # mutually exclusive in Cosyne's own constructor -- exactly one of the two may be
    # non-None at a time. ----
    "cosyne_tournament_size": 4,
    "cosyne_mutation_stdev": 0.3,
    "cosyne_mutation_probability": 0.5,
    "cosyne_permute_all": True,
    "cosyne_num_elites": 1,
    "cosyne_elitism_ratio": None,
    "cosyne_eta": None,
    "cosyne_num_children": None,

    # ---- general experiment parameters ----
    "n_neurons": 15,
    "evo_context_cues_on": False,
    "evo_sensory_cues_on": True,
    "evo_plasticity_on": True,
    "paradigm": "trainA, 50, replay, 10, trainB, 50",
    "dt": 0.2,
    "tau": 1.0,
    "noise_std": 0.1,
    "straight_thresh": 1 / 3,
    "big_reward": 1.0,
    "small_reward": 0.0,
    "crash_penalty": -0.4,
    "turn_reward_big": 1.0,
    "turn_reward_small": 0.0,
    "l1_lambda": 0.001,

    # ---- anything else (tracking/plot cadence) ----
    "tracked_per_interval": 50,
    "max_networks_preview": 6,
    "max_runs_preview": 20,
    "hist_bin_width": 1,
}

# applied on top of BASE_CONFIG for this batch, before SWEEP_PARAMS -- so a swept
# param always wins over an override on the same key
OVERRIDES = {}


# ==== 3. CONFIG GENERATION FUNCTIONS ========================================
def _sweep_combinations(sweep_params):
    """Cartesian product of sweep_params' value lists -> list of {param: value}
    dicts, one per combination. Blind to how many params/values are given."""
    names = list(sweep_params.keys())
    value_lists = [sweep_params[name] for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def _combo_slug(combo):
    """Turn a {param: value} combination into a filesystem-safe filename fragment,
    e.g. {"n_neurons": 9, "evo_context_cues_on": True} -> 'n_neurons_9_evo_context_cues_on_True'."""
    return "_".join(f"{name}_{value}" for name, value in combo.items())


def _generate_seed_list(mode, n, fixed_values, seed_name):
    """One list of n seeds for seed_name, per SEED_MODE."""
    if mode == "random":
        return random.sample(range(0, 100_000), n)
    if mode == "incrementing":
        return list(range(n))
    if mode == "fixed":
        if len(fixed_values) != n:
            raise ValueError(f"FIXED_SEEDS['{seed_name}'] has {len(fixed_values)} entries, need {n}.")
        return list(fixed_values)
    raise ValueError(f"Unknown SEED_MODE '{mode}'; valid values are 'random', 'incrementing', 'fixed'.")


def build_config(combo, seeds, base_config=BASE_CONFIG, overrides=OVERRIDES):
    """Merge base_config + overrides + this combination's swept params + this config's seeds."""
    return {**base_config, **overrides, **combo, **seeds}


def write_config_batch(sweep_params=SWEEP_PARAMS, n_configs_per_group=N_CONFIGS_PER_GROUP,
                        seed_mode=SEED_MODE, same_randoms_per_group=SAME_RANDOMS_PER_GROUP,
                        fixed_seeds=FIXED_SEEDS, output_folder=OUTPUT_FOLDER,
                        filename_prefix=FILENAME_PREFIX, base_config=BASE_CONFIG, overrides=OVERRIDES):
    """Write one config JSON per (parameter combination x seeded config) into output_folder."""
    output_folder.mkdir(parents=True, exist_ok=True)

    combinations = _sweep_combinations(sweep_params)
    # same_randoms_per_group: one seed list of length n_configs_per_group, reused
    # (re-indexed by i, not global_idx) for every combination. Otherwise every
    # config across the whole batch gets its own independently-generated seed.
    seed_list_len = n_configs_per_group if same_randoms_per_group else len(combinations) * n_configs_per_group
    seed_names = ["master_seed", "noise_seed", "reward_seed", "test_seed", "weight_init_seed"]
    seed_lists = {
        name: _generate_seed_list(seed_mode, seed_list_len, fixed_seeds.get(name, []), name)
        for name in seed_names
    }

    written_paths = []
    global_idx = 0
    for combo in combinations:
        slug = _combo_slug(combo)
        for i in range(n_configs_per_group):
            seed_idx = i if same_randoms_per_group else global_idx
            seeds = {name: seed_lists[name][seed_idx] for name in seed_names}
            config = build_config(combo, seeds, base_config, overrides)
            path = output_folder / f"{filename_prefix}{slug}_{i}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            written_paths.append(path)
            global_idx += 1

    return written_paths


# ==== 4. MAIN EXECUTION =====================================================
if __name__ == "__main__":
    paths = write_config_batch()
    n_combinations = len(_sweep_combinations(SWEEP_PARAMS))
    print(f"Wrote {len(paths)} config file(s) across {n_combinations} parameter "
          f"combination(s) ({N_CONFIGS_PER_GROUP} configs each) to {OUTPUT_FOLDER}")
    for path in paths:
        print(f"  {path.name}")
