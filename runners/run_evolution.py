"""Run PGPE with tracking output and save end-of-run plots + full numeric results.

Usage: python run_evolution.py <config_name> <experiment_name> <device> <chain_label> <early_termination_enabled>

chain_label is just a display label for terminal output (run_batch.py passes its chain
index; a manual run can pass anything, e.g. "manual").

early_termination_enabled is exactly "True" or "False" (fails loudly on anything else -
argv values are always strings, so this project's style forbids silently guessing what a
different value would mean). When "True" (the normal case): on a detailed-print
generation (skipping generation 1), for EITHER task (trainA/trainB) across the whole
tracked population, if one turn direction (or one CORRECT turn direction) never happened
at all, the run stops there and saves everything normally, as if that were the final
generation - the population has collapsed onto a degenerate policy that isn't going
to develop further, so finishing out the configured generation count is wasted
compute. See sim_core/fitness.py's _check_event_count_termination for the exact
conditions (and why "every run crashed" is deliberately NOT one of them). When "False":
the same criterion is still detected and printed every time it's met, but the run is
never actually stopped early - it always runs the full configured generation count -
for deliberately forcing a run past what looks like a collapsed population, e.g. to see
whether it recovers given more generations.
"""

# ==== 1) RNG DETERMINISM + PATH SETUP ==========================================
import os 

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import datetime
import functools
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

import evotorch
import matplotlib
matplotlib.use("Agg")  # headless - this only ever calls savefig(), never plt.show(), and a
                        # remote GPU server reached over VPN may have no display/Tk at all
import matplotlib.pyplot as plt
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import PGPE
from evotorch.tools import stdev_from_radius

from analysis.decision_plotting import (
    DECISION_CMAP,
    DECISION_COLORS,
    DECISION_LABELS,
    N_DECISION_CATEGORIES,
    decision_category_matrix,
    draw_decisions_panel,
    grid_dims,
    sort_by_fitness,
)
from analysis.csv_export import write_csv
from analysis.results_io import results_filename, save_results_h5
from sim_core import constants, genome_codec
from sim_core.constants import INPUT_SENSORY_A, INPUT_SENSORY_B
from sim_core.fitness import PHASE_CONTEXT, configure_printing, fitness_function, get_printing_history, should_terminate_early
from sim_core.paradigm import PHASE_REPLAY, parse_paradigm

# ==== 2) CONFIG LOADING + OUTPUT LOCATION =======================================
# Results always live under the hardcoded DATA_ROOT - not a user choice. The
# config is found by joining CONFIGS_ROOT with the given name + ".json" - that's
# it, nothing else: for a plain name that's configs/<name>.json; run_batch.py
# reaches configs/batch_to_run/<name>.json the same way, by passing
# "batch_to_run/<name>" as that same argument. No defaults/fallbacks on the
# config contents - a missing or malformed field fails loudly (KeyError), on purpose.
CONFIGS_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"


def _parse_bool_arg(value, arg_name):
    """Strict True/False CLI-argument parser - argv values are always strings, and this
    project's style forbids silently guessing (e.g. treating any non-"False" string as
    True), so anything other than exactly "True" or "False" fails loudly."""
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{arg_name} must be exactly 'True' or 'False', got {value!r}")


if len(sys.argv) != 6:
    raise ValueError(
        "Usage: python run_evolution.py <config_name> <experiment_name> <device> <chain_label> "
        "<early_termination_enabled: True/False>"
    )
CONFIG_PATH = CONFIGS_ROOT / f"{sys.argv[1]}.json"
OUTPUT_ROOT = DATA_ROOT / sys.argv[2]
CHAIN_LABEL = sys.argv[4]  # printed as "CHAIN <label>" in terminal output - run_batch.py passes
                            # its chain index; a manual run can pass anything, e.g. "manual"
EARLY_TERMINATION_ENABLED = _parse_bool_arg(sys.argv[5], "early_termination_enabled")
with open(CONFIG_PATH, "r", encoding="utf-8") as _config_file:
    CONFIG = json.load(_config_file)

RUN_NAME = CONFIG_PATH.stem  # output folder is always named after the input config file
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # year->...->second, so
                                                                     # alphabetical (file explorer)
                                                                     # order is chronological order
RUN_DIR = OUTPUT_ROOT / f"{RUN_NAME}_{RUN_TIMESTAMP}"  # everything this run produces lives here
DEVICE = sys.argv[3]  # required CLI input - always wins, even if the config json still has its
                       # own (by-now-vestigial) "device" field
N_NEURONS = CONFIG["n_neurons"]  # total neurons; see constants.configure_network() for the fixed
                                  # input/output assignment + derived hidden-neuron count
MASTER_SEED = CONFIG["master_seed"]
NOISE_SEED = CONFIG["noise_seed"]
REWARD_SEED = CONFIG["reward_seed"]
TEST_SEED = CONFIG["test_seed"]  # dedicated RNG stream for cue-importance measurement only - never
                                  # touches noise_generator/reward_generator, so the tracking interval
                                  # can't change the run
WEIGHT_INIT_SEED = CONFIG["weight_init_seed"]  # dedicated RNG stream for the fresh per-lifetime initial-weight
                                                # draw (sample_initial_weights) - never touches any other stream

EVO_CONTEXT_CUES_ON = CONFIG["evo_context_cues_on"]  # if False, context-cue input neurons are clipped to zero during evolution
EVO_SENSORY_CUES_ON = CONFIG["evo_sensory_cues_on"]  # if False, sensory-cue input neurons are clipped to zero during evolution
EVO_PLASTICITY_ON = CONFIG["evo_plasticity_on"]  # if False, eta is forced to all zeros regardless of genome -> no plasticity, frozen weights

# Per-evaluation phase sequence: comma-separated (phase, value) pairs, where phase
# is one of "trainA"/"trainB" (value = number of maze runs) or "replay" (value =
# number of ticks). Parsed eagerly below so a malformed string fails at import time.
PARADIGM = CONFIG["paradigm"]
PARADIGM_PHASES = parse_paradigm(PARADIGM)
# ordered list of (phase_type, num_runs) for training phases only, replay skipped -
# this must stay in the same order fitness.py concatenates tracking segments in
TRAINING_PHASE_LAYOUT = [(phase_type, value) for phase_type, value in PARADIGM_PHASES if phase_type != PHASE_REPLAY]

# FWT/BWT (see sim_core/fitness.py's _measure_transfer_metrics) generalize to however
# many DISTINCT training tasks this paradigm actually has (T >= 2, in whatever order the
# paradigm trains them) - today T is always 2, but nothing here (or in fitness.py) needs
# touching if a future paradigm adds a third task, e.g. a double-T-maze - only
# paradigm.py's VALID_PHASE_TYPES and fitness.py's PHASE_CONTEXT need to learn the new
# task type exists at all. Fail loudly here, at config-load time, if the paradigm has
# fewer than two distinct training tasks or repeats one - rather than only discovering
# it once fitness.py's own identical check fires mid-evolution. TASK_ORDER_LABELS (e.g.
# ["A", "B"] or, one day, ["A", "B", "C"]) reflects this run's ACTUAL paradigm order, so
# plot/CSV labels are always correct regardless of task count or order.
_TRAINING_TASK_TYPES = [phase_type for phase_type, _ in TRAINING_PHASE_LAYOUT]
if len(_TRAINING_TASK_TYPES) < 2 or len(set(_TRAINING_TASK_TYPES)) != len(_TRAINING_TASK_TYPES):
    raise ValueError(
        "Forward/backward transfer metrics (see sim_core/fitness.py's _measure_transfer_metrics) "
        f"need at least two DISTINCT training tasks, each appearing exactly once; got "
        f"{TRAINING_PHASE_LAYOUT} from paradigm {PARADIGM!r}."
    )
TASK_ORDER_LABELS = [PHASE_CONTEXT[phase_type] for phase_type in _TRAINING_TASK_TYPES]

NUM_GENERATIONS = CONFIG["num_generations"]
SEARCH_POPSIZE = CONFIG["search_popsize"]
RADIUS_INIT = CONFIG["radius_init"]     # radius of the initial search hypersphere in genome space (GENOME_LENGTH-dim), sweep/ optimize
MAX_SPEED = RADIUS_INIT / 15.0          # evotorch's rule of thumb from the ClipUp paper: max_speed = radius / 15.0, adjust the 15.0 to optimize
CENTER_LEARNING_RATE = MAX_SPEED / 2    # this is the step size in the ClipUp paper
STDEV_LEARNING_RATE = CONFIG["stdev_learning_rate"]
MOMENTUM = CONFIG["momentum"]


def _config_get_if_enabled(enabled, key):
    """Reads CONFIG[key], but only requires it to be present if `enabled` is True.

    These pgpe_* anti-stagnation keys (added 2026-08) are opt-in and postdate ~500
    existing sweep/archive configs under configs/ that were never meant to define them.
    Retrofitting a required key onto every one of those was judged out of scope for
    "implement these 3 measures" and risked touching other queued/archived experiments.
    So: when a measure is off, its parameters are allowed to be absent (CONFIG.get(...)
    with a None default) and are never read. When a measure IS turned on for a given
    run's config, its parameters go back to this project's normal fail-loudly rule -
    a missing key still raises KeyError via plain CONFIG[key] indexing, same as every
    other field above."""
    if not enabled:
        return None
    return CONFIG[key]


# ---- PGPE anti-stagnation measures (all opt-in, off unless enabled in this run's config) ----
# All three are independently toggleable via their own "*_enabled" flag, specifically so they
# can be turned on together for an initial "big swing" test and then removed one at a time to
# see which one(s) actually mattered.

PGPE_STDEV_MIN_ENABLED = CONFIG.get("pgpe_stdev_min_enabled")
# Elementwise floor on PGPE's search stdev, enforced natively by evotorch's PGPE/
# GaussianSearchAlgorithm on every generation (see stdev_min passed into PGPE(...) below) -
# NOT a manual post-hoc clamp. Keeps the search distribution from ever collapsing its
# exploration width below this value in any genome dimension.
PGPE_STDEV_MIN = _config_get_if_enabled(PGPE_STDEV_MIN_ENABLED, "pgpe_stdev_min")

PGPE_RESTART_ENABLED = CONFIG.get("pgpe_restart_enabled")
# Stagnation-triggered restart: if the population-best fitness hasn't improved by more than
# pgpe_restart_min_improvement for pgpe_restart_patience consecutive generations, stdev is
# reset to the radius given by pgpe_restart_radius (converted the same way radius_init is,
# via evotorch's stdev_from_radius) and the ClipUp momentum buffer is zeroed. Center is left
# untouched.
PGPE_RESTART_PATIENCE = _config_get_if_enabled(PGPE_RESTART_ENABLED, "pgpe_restart_patience")
PGPE_RESTART_MIN_IMPROVEMENT = _config_get_if_enabled(PGPE_RESTART_ENABLED, "pgpe_restart_min_improvement")
PGPE_RESTART_RADIUS = _config_get_if_enabled(PGPE_RESTART_ENABLED, "pgpe_restart_radius")

PGPE_CENTER_PERTURB_ENABLED = CONFIG.get("pgpe_center_perturb_enabled")
# Every pgpe_center_perturb_interval generations, adds isolated N(0, pgpe_center_perturb_std)
# noise directly onto the search distribution's center - independent of the restart trigger,
# meant to nudge PGPE off flat/plateau regions even when stdev hasn't collapsed enough to
# fire a restart. Uses its own seeded RNG stream (pgpe_perturb_seed).
PGPE_CENTER_PERTURB_INTERVAL = _config_get_if_enabled(PGPE_CENTER_PERTURB_ENABLED, "pgpe_center_perturb_interval")
PGPE_CENTER_PERTURB_STD = _config_get_if_enabled(PGPE_CENTER_PERTURB_ENABLED, "pgpe_center_perturb_std")
PGPE_PERTURB_SEED = _config_get_if_enabled(PGPE_CENTER_PERTURB_ENABLED, "pgpe_perturb_seed")

L1_LAMBDA = CONFIG["l1_lambda"]

TRACKED_PER_INTERVAL = CONFIG["tracked_per_interval"]
MAX_NETWORKS_PREVIEW = CONFIG["max_networks_preview"]
MAX_RUNS_PREVIEW = CONFIG["max_runs_preview"]
HIST_BIN_WIDTH = CONFIG["hist_bin_width"]
PLOT_DPI = 180 

# Network layout + tunable sim_core constants (reward shaping + CTRNN dynamics) -
# set once, here, before any simulation code runs; sim_core modules read
# constants.X live at call time, so this is the only place that needs to know
# about the config file. configure_network() must run before genome_codec's
# functions are called (they size tensors from constants.N).
constants.configure_network(n_neurons=N_NEURONS)
constants.configure(
    dt=CONFIG["dt"],
    tau=CONFIG["tau"],
    noise_std=CONFIG["noise_std"],
    straight_thresh=CONFIG["straight_thresh"],
    big_reward=CONFIG["big_reward"],
    small_reward=CONFIG["small_reward"],
    crash_penalty=CONFIG["crash_penalty"],
    turn_reward_big=CONFIG["turn_reward_big"],
    turn_reward_small=CONFIG["turn_reward_small"],
)

GENOME_SPEC = genome_codec.genome_spec()
GENOME_LENGTH = genome_codec.genome_length()

DECISIONS_FILENAME = "decisions.png"
ALL_DECISIONS_FILENAME = "all_decisions.png"
EVENT_COUNTS_FILENAME = "event_counts.png"
REWARD_HIST_FILENAME = "reward_hist.png"
FROBENIUS_FILENAME = "frobenius.png"
WEIGHT_DISTRIBUTION_FILENAME = "weight_distribution.png"
REWARD_EVOLUTION_FILENAME = "reward_evolution.png"
SENSORY_CUE_FILENAME = "sensory_cues.png"
TRAINING_REWARD_FILENAME = "training_reward_evolution.png"
L1_EVOLUTION_FILENAME = "l1_evolution.png"
INPUT_WEIGHING_FILENAME = "input_weighing.png"
TRANSFER_METRICS_FILENAME = "transfer_metrics.png"
PGPE_PARAMS_FILENAME = "pgpe_params.png"
PGPE_FITNESS_FILENAME = "pgpe_fitness.png"
PGPE_STAGNATION_FILENAME = "pgpe_stagnation.png"

# One CSV companion per PNG above (same stem, ".csv" instead of ".png") - holds
# exactly the already-wrangled data that PNG was drawn from, see analysis/csv_export.py.
DECISIONS_CSV_FILENAME = "decisions.csv"
ALL_DECISIONS_CSV_FILENAME = "all_decisions.csv"
EVENT_COUNTS_CSV_FILENAME = "event_counts.csv"
REWARD_HIST_CSV_FILENAME = "reward_hist.csv"
FROBENIUS_CSV_FILENAME = "frobenius.csv"
WEIGHT_DISTRIBUTION_CSV_FILENAME = "weight_distribution.csv"
REWARD_EVOLUTION_CSV_FILENAME = "reward_evolution.csv"
SENSORY_CUE_CSV_FILENAME = "sensory_cues.csv"
TRAINING_REWARD_CSV_FILENAME = "training_reward_evolution.csv"
L1_EVOLUTION_CSV_FILENAME = "l1_evolution.csv"
INPUT_WEIGHING_CSV_FILENAME = "input_weighing.csv"
TRANSFER_METRICS_CSV_FILENAME = "transfer_metrics.csv"
PGPE_PARAMS_CSV_FILENAME = "pgpe_params.csv"
PGPE_FITNESS_CSV_FILENAME = "pgpe_fitness.csv"
PGPE_STAGNATION_CSV_FILENAME = "pgpe_stagnation.csv"
REWARD_EVOLUTION_COLORS = ["#E07A5F", "#3D405B", "#81B29A"]
PALETTE_COLORS = ["#E07A5F", "#3D405B", "#81B29A", "#F2CC8F", "#F4F1DE"]
WEIGHT_HIST_BINS = 80

# Decision-outcome color/label scheme (DECISION_COLORS/LABELS/N_DECISION_CATEGORIES)
# lives in analysis/decision_plotting.py now, shared with run_batch.py's facet plots.


def _validate_device_or_raise(device):
    if device != "cuda":
        return
    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "Config requests device='cuda', but PyTorch CUDA is unavailable in this environment. "
        f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
        f"cuda_device_count={torch.cuda.device_count()}. "
        "Use a driver/runtime compatible with this PyTorch wheel, install a matching PyTorch build, "
        "or change the config device to 'cpu'."
    )


# ==== 3) PGPE DIAGNOSTIC TRACKING ==============================================
_SENSORY_CUE_NEURON_INDICES = (INPUT_SENSORY_A, INPUT_SENSORY_B)


def _sensory_cue_genome_indices():
    """Flat-genome indices touching sensory-cue neurons in any N-sized axis."""
    indices = []
    offset = 0
    for _, shape in GENOME_SPEC:
        for local_idx in np.ndindex(shape):
            if any(
                (axis_size > max(_SENSORY_CUE_NEURON_INDICES)) and (axis_value in _SENSORY_CUE_NEURON_INDICES)
                for axis_size, axis_value in zip(shape, local_idx)
            ):
                indices.append(offset + int(np.ravel_multi_index(local_idx, shape)))
        offset += int(np.prod(shape))
    return torch.tensor(indices, dtype=torch.long)


SENSORY_CUE_GENOME_INDICES = _sensory_cue_genome_indices()


def _init_pgpe_history():
    return {
        "generation": [],
        "center_norm": [],
        "stdev_mean": [],
        "stdev_min": [],
        "stdev_max": [],
        "stdev_sensory_mean": [],
        "stdev_sensory_min": [],
        "stdev_sensory_max": [],
        "fitness_mean": [],
        "fitness_max": [],
        "fitness_std": [],
        # Anti-stagnation event log - meaningful only for whichever of the three measures
        # is enabled in this run's config; 0/False throughout when a measure is disabled.
        "generations_since_improvement": [],
        "restart_triggered": [],
        "center_perturbed": [],
    }


def _collect_pgpe_history(
    searcher,
    reward_evolution,
    pgpe_history,
    *,
    generations_since_improvement=0,
    restart_triggered=False,
    center_perturbed=False,
):
    status = searcher.status

    generation = int(reward_evolution["generation"][-1])
    center = status.get("center", None)
    if center is None:
        center = getattr(searcher, "center", None)
    stdev = status.get("stdev", None)
    if stdev is None:
        stdev = getattr(searcher, "stdev", None)
    if center is None or stdev is None:
        raise RuntimeError(
            "PGPE tracking could not find PGPE center/stdev in searcher.status or as searcher attributes."
        )

    center = center.detach().reshape(-1).float().cpu()
    stdev = stdev.detach().reshape(-1).float().cpu()
    if SENSORY_CUE_GENOME_INDICES.numel() == 0:
        raise RuntimeError("Sensory-cue index set is empty.")
    sensory_stdev = stdev[SENSORY_CUE_GENOME_INDICES]

    pgpe_history["generation"].append(generation)
    pgpe_history["center_norm"].append(float(torch.linalg.vector_norm(center).item()))
    pgpe_history["stdev_mean"].append(float(stdev.mean().item()))
    pgpe_history["stdev_min"].append(float(stdev.min().item()))
    pgpe_history["stdev_max"].append(float(stdev.max().item()))
    pgpe_history["stdev_sensory_mean"].append(float(sensory_stdev.mean().item()))
    pgpe_history["stdev_sensory_min"].append(float(sensory_stdev.min().item()))
    pgpe_history["stdev_sensory_max"].append(float(sensory_stdev.max().item()))
    pgpe_history["fitness_mean"].append(float(reward_evolution["mean_eval"][-1]))
    pgpe_history["fitness_max"].append(float(reward_evolution["pop_best_eval"][-1]))
    pgpe_history["fitness_std"].append(float(reward_evolution["std_eval"][-1]))
    pgpe_history["generations_since_improvement"].append(int(generations_since_improvement))
    pgpe_history["restart_triggered"].append(bool(restart_triggered))
    pgpe_history["center_perturbed"].append(bool(center_perturbed))


def _restart_pgpe_distribution(searcher):
    """Stagnation-triggered "big swing" restart: re-inflate the search distribution's
    stdev back to PGPE_RESTART_RADIUS (converted to stdev via evotorch's own
    stdev_from_radius, the same conversion used for radius_init at construction time),
    and zero out the ClipUp optimizer's momentum buffer.

    Center is deliberately left untouched here - a restart hands PGPE's current
    best-guess center a fresh, wide search radius to explore around, rather than
    discarding progress already made.

    Momentum is reset alongside stdev because stale ClipUp velocity would immediately 
    drag the center back along the pre-restart trajectory on the very next step, 
    undermining the point of re-inflating exploration.

    Mutates searcher._distribution.sigma directly (verified against evotorch 0.6.1's
    SeparableGaussian: mu/sigma are plain settable properties on the *live* distribution
    object, re-read fresh by _fill()/update_parameters() on every subsequent step - there
    is no other cached state to go stale). searcher.center/.stdev are read-only status-getter
    proxies onto this same object and are NOT safe to assign directly - doing so would
    silently create a shadow attribute that the actual search loop never reads.
    """
    restart_sigma_value = stdev_from_radius(PGPE_RESTART_RADIUS, GENOME_LENGTH)
    new_sigma = torch.full((GENOME_LENGTH,), restart_sigma_value, device=DEVICE)
    if PGPE_STDEV_MIN_ENABLED:
        # Don't restart to below the floor if the floor (independently) exceeds the
        # restart radius' stdev - keeps the two measures from fighting each other.
        new_sigma = torch.clamp(new_sigma, min=PGPE_STDEV_MIN)
    searcher._distribution.sigma = new_sigma

    if searcher.optimizer is not None:
        searcher.optimizer._velocity.zero_()


def _perturb_pgpe_center(searcher, generator):
    """Adds one isolated N(0, PGPE_CENTER_PERTURB_STD) draw per genome dimension directly
    onto the search distribution's center. Same mutation-safety
    reasoning as _restart_pgpe_distribution applies here to searcher._distribution.mu."""
    current_mu = searcher._distribution.mu
    noise = torch.randn(current_mu.shape, generator=generator, device=DEVICE) * PGPE_CENTER_PERTURB_STD
    searcher._distribution.mu = current_mu + noise


def _save_pgpe_params_plot(plot_dir, pgpe_history):
    generations = np.array(pgpe_history["generation"])
    center_norm = np.array(pgpe_history["center_norm"])
    stdev_mean = np.array(pgpe_history["stdev_mean"])
    stdev_min = np.array(pgpe_history["stdev_min"])
    stdev_max = np.array(pgpe_history["stdev_max"])
    stdev_sensory_mean = np.array(pgpe_history["stdev_sensory_mean"])
    stdev_sensory_min = np.array(pgpe_history["stdev_sensory_min"])
    stdev_sensory_max = np.array(pgpe_history["stdev_sensory_max"])

    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), dpi=PLOT_DPI)
    axes[0].plot(generations, center_norm, color="#3D405B", linewidth=2.0)
    axes[0].set_title("PGPE center norm")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("L2 norm")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(generations, stdev_mean, color="#81B29A", linewidth=2.0, label="mean")
    axes[1].plot(generations, stdev_min, color="#E07A5F", linewidth=1.5, label="min")
    axes[1].plot(generations, stdev_max, color="#3D405B", linewidth=1.5, label="max")
    axes[1].set_title("PGPE stdev (all dims)")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("stdev")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend()

    axes[2].plot(generations, stdev_sensory_mean, color="#81B29A", linewidth=2.0, label="mean")
    axes[2].plot(generations, stdev_sensory_min, color="#E07A5F", linewidth=1.5, label="min")
    axes[2].plot(generations, stdev_sensory_max, color="#3D405B", linewidth=1.5, label="max")
    axes[2].set_title("PGPE stdev (sensory-cue dims)")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("stdev")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, PGPE_PARAMS_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, PGPE_PARAMS_CSV_FILENAME), _columns_to_rows({
        "generation": generations,
        "center_norm": center_norm,
        "stdev_mean": stdev_mean, "stdev_min": stdev_min, "stdev_max": stdev_max,
        "stdev_sensory_mean": stdev_sensory_mean, "stdev_sensory_min": stdev_sensory_min, "stdev_sensory_max": stdev_sensory_max,
    }))


def _save_pgpe_fitness_plot(plot_dir, pgpe_history):
    generations = np.array(pgpe_history["generation"])
    fitness_mean = np.array(pgpe_history["fitness_mean"])
    fitness_max = np.array(pgpe_history["fitness_max"])
    fitness_std = np.array(pgpe_history["fitness_std"])

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=PLOT_DPI)
    axes[0].plot(generations, fitness_mean, color="#81B29A", linewidth=2.0, label="mean")
    axes[0].plot(generations, fitness_max, color="#3D405B", linewidth=2.0, label="max")
    axes[0].set_title("Sampled population fitness")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Fitness")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    axes[1].plot(generations, fitness_std, color="#E07A5F", linewidth=2.0)
    axes[1].set_title("Sampled population fitness std")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Fitness std")
    axes[1].grid(True, alpha=0.2)

    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, PGPE_FITNESS_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, PGPE_FITNESS_CSV_FILENAME), _columns_to_rows({
        "generation": generations, "fitness_mean": fitness_mean, "fitness_max": fitness_max, "fitness_std": fitness_std,
    }))


def _save_pgpe_stagnation_plot(plot_dir, pgpe_history):
    """Save a 3-panel diagnostic figure for the anti-stagnation measures (see the
    "PGPE anti-stagnation measures" section above): fitness vs. restart/perturb events
    (top), search stdev vs. its configured floor (middle), and the stagnation counter vs.
    its configured patience with restart/perturb event markers (bottom). Purely a new
    visualization of pgpe_history columns _collect_pgpe_history already populates every
    generation -- renders sensibly even when all three measures are disabled for this run
    (then the event markers/reference lines just never appear, which is itself informative)."""
    generations = np.array(pgpe_history["generation"])
    fitness_max = np.array(pgpe_history["fitness_max"])
    fitness_mean = np.array(pgpe_history["fitness_mean"])
    stdev_mean = np.array(pgpe_history["stdev_mean"])
    generations_since_improvement = np.array(pgpe_history["generations_since_improvement"])
    restart_triggered = np.array(pgpe_history["restart_triggered"], dtype=bool)
    center_perturbed = np.array(pgpe_history["center_perturbed"], dtype=bool)

    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(13, 10), dpi=PLOT_DPI, sharex=True)

    # ---- panel 1: fitness vs. restart/perturb events ----
    axis = axes[0]
    axis.plot(generations, fitness_max, color="#3D405B", linewidth=2.0, label="fitness_max")
    axis.plot(generations, fitness_mean, color="#81B29A", linewidth=1.2, label="fitness_mean")
    for gen in generations[restart_triggered]:
        axis.axvline(gen, color="#E07A5F", alpha=0.55, linewidth=1.0)
    for gen in generations[center_perturbed]:
        axis.axvline(gen, color="#888888", alpha=0.35, linewidth=0.6)
    axis.set_title(f"{RUN_NAME} ({RUN_TIMESTAMP}): fitness vs. restart/perturb events", loc="left")
    axis.set_ylabel("Fitness")
    axis.grid(True, alpha=0.2)
    axis.legend()

    # ---- panel 2: stdev_mean vs. configured floor (reference line only if the floor is
    # actually enabled for this run -- drawing it at a placeholder value would mislead) ----
    axis = axes[1]
    axis.plot(generations, stdev_mean, color="#3D405B", linewidth=2.0)
    y_upper = stdev_mean.max()
    if PGPE_STDEV_MIN_ENABLED:
        axis.axhline(PGPE_STDEV_MIN, color="#888888", linewidth=1.2, linestyle="--", label=f"configured floor ({PGPE_STDEV_MIN})")
        y_upper = max(y_upper, PGPE_STDEV_MIN)
        axis.legend()
    axis.set_ylim(0, y_upper * 1.1)
    axis.set_title("Search-distribution stdev vs. configured floor")
    axis.set_ylabel("stdev")
    axis.grid(True, alpha=0.2)

    # ---- panel 3: stagnation counter + when each measure fired (same "skip the reference
    # line if disabled" reasoning as panel 2 applies to the patience line here) ----
    axis = axes[2]
    axis.plot(generations, generations_since_improvement, color="#3D405B", linewidth=1.5, label="generations_since_improvement")
    if PGPE_RESTART_ENABLED:
        axis.axhline(PGPE_RESTART_PATIENCE, color="#888888", linewidth=1.2, linestyle=":", label=f"patience ({PGPE_RESTART_PATIENCE})")
    if restart_triggered.any():
        axis.scatter(generations[restart_triggered], np.full(restart_triggered.sum(), -1.0), color="#E07A5F", s=20, label="restart")
    if center_perturbed.any():
        axis.scatter(generations[center_perturbed], np.full(center_perturbed.sum(), -2.5), color="#888888", s=20, marker="|", label="perturb")
    y_upper = PGPE_RESTART_PATIENCE + 2 if PGPE_RESTART_ENABLED else int(generations_since_improvement.max()) + 2
    axis.set_ylim(-4, y_upper)
    axis.set_xlabel("Generation")
    axis.set_ylabel("Generations since improvement")
    axis.grid(True, alpha=0.2)
    axis.legend()

    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, PGPE_STAGNATION_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, PGPE_STAGNATION_CSV_FILENAME), _columns_to_rows({
        "generation": generations, "fitness_max": fitness_max, "fitness_mean": fitness_mean, "stdev_mean": stdev_mean,
        "generations_since_improvement": generations_since_improvement,
        "restart_triggered": pgpe_history["restart_triggered"], "center_perturbed": pgpe_history["center_perturbed"],
    }))


# ==== 4) PLOTTING HELPERS ======================================================
def _prefixed_path(plot_dir, filename):
    """Prefix every saved figure's filename with RUN_NAME, e.g. 'decisions.png' -> 'NAME_decisions.png'."""
    return plot_dir / f"{RUN_NAME}_{filename}"


def _columns_to_rows(columns):
    """columns: dict of column_name -> equal-length sequence (first column is normally
    "generation"). Returns one dict (row) per index, column order matching insertion
    order - for the many per-generation line plots below whose CSV is just their
    plotted line(s) transposed into rows."""
    names = list(columns.keys())
    length = len(columns[names[0]])
    return [{name: columns[name][i] for name in names} for i in range(length)]


def _generation_colors(tracked_generations):
    """Create light-gray to black colors for tracked generations."""
    count = len(tracked_generations)
    gray_values = np.linspace(0.8, 0.0, count)
    colors = []
    for gray in gray_values:
        colors.append((gray, gray, gray, 1.0))
    return colors


def _record_sorted_by_fitness(record):
    """Return a copy of record with all per-network arrays sorted best-to-worst by fitness."""
    (order,) = sort_by_fitness(record["fitness"].numpy())
    sorted_record = dict(record)
    for key in ("decisions_by_run", "crashed_by_run", "rewarded_by_run",
                "big_reward_by_run", "sensory_cue_by_run", "correct_arm_by_run"):
        sorted_record[key] = record[key][order]
    return sorted_record


def _record_decision_matrix(record):
    """Encode record's per-run outcomes into category ids via the shared encoder
    (see analysis/decision_plotting.py's DECISION_LABELS)."""
    return decision_category_matrix(
        record["decisions_by_run"].numpy(),
        record["crashed_by_run"].numpy(),
        record["rewarded_by_run"].numpy(),
        record["correct_arm_by_run"].numpy(),
    )


def _decision_rows(generation, matrix):
    """Long-format rows for one generation's fitness-sorted decision matrix ([network_rank,
    run_index] category codes): one row per cell, network_rank 0 = best fitness."""
    rows = []
    n_networks, n_runs = matrix.shape
    for rank in range(n_networks):
        for run_index in range(n_runs):
            code = int(matrix[rank, run_index])
            rows.append({
                "generation": generation,
                "network_rank": rank,
                "run_index": run_index,
                "category_code": code,
                "category_label": DECISION_LABELS[code],
            })
    return rows


def _save_decisions_plot(plot_dir, tracked_records):
    """Save side-by-side heatmaps for first and last tracked generations, sorted by fitness."""
    first_matrix = _record_decision_matrix(_record_sorted_by_fitness(tracked_records[0]))
    last_matrix = _record_decision_matrix(_record_sorted_by_fitness(tracked_records[-1]))

    figure = plt.figure(figsize=(18, 9), dpi=PLOT_DPI)
    grid = figure.add_gridspec(nrows=2, ncols=2, height_ratios=[20, 1], hspace=0.28, wspace=0.12)
    ax0 = figure.add_subplot(grid[0, 0])
    ax1 = figure.add_subplot(grid[0, 1])
    colorbar_axis = figure.add_subplot(grid[1, :])

    im0 = draw_decisions_panel(ax0, first_matrix, f"Generation {tracked_records[0]['generation']}")
    draw_decisions_panel(ax1, last_matrix, f"Generation {tracked_records[-1]['generation']}")

    colorbar = figure.colorbar(im0, cax=colorbar_axis, orientation="horizontal", ticks=np.arange(0, N_DECISION_CATEGORIES, 1))
    colorbar.ax.set_xticklabels(DECISION_LABELS)
    figure.suptitle("Decisions: first vs last tracked generation (sorted by fitness)")
    # bbox_inches="tight" at save time instead of figure.tight_layout(): tight_layout()
    # doesn't support the colorbar's gridspec-placed Axes and warns every run (same
    # approach _save_all_decisions_plot below already uses for the same reason)
    figure.savefig(_prefixed_path(plot_dir, DECISIONS_FILENAME), bbox_inches="tight")
    plt.close(figure)

    rows = (
        _decision_rows(tracked_records[0]["generation"], first_matrix)
        + _decision_rows(tracked_records[-1]["generation"], last_matrix)
    )
    write_csv(_prefixed_path(plot_dir, DECISIONS_CSV_FILENAME), rows)


def _save_all_decisions_plot(plot_dir, tracked_records, tracked_generations):
    """Save a grid of decision heatmaps for every tracked generation, sorted by fitness."""
    n_plots = len(tracked_records)
    n_rows, n_cols = grid_dims(n_plots)

    # Fixed panel size in inches so labels always look the same regardless of grid size.
    panel_w = 8.0
    panel_h = 6.0
    colorbar_h = 0.7
    title_h = 0.5
    fs_title = 14
    fs_axis = 11
    fs_colorbar = 12

    fig_w = panel_w * n_cols
    fig_h = panel_h * n_rows + colorbar_h + title_h

    figure = plt.figure(figsize=(fig_w, fig_h), dpi=PLOT_DPI)
    grid = figure.add_gridspec(
        nrows=n_rows + 1, ncols=n_cols,
        height_ratios=[panel_h] * n_rows + [colorbar_h],
        hspace=0.45, wspace=0.18,
    )

    im_ref = None
    csv_rows = []
    for idx, record in enumerate(tracked_records):
        row, col = divmod(idx, n_cols)
        matrix = _record_decision_matrix(_record_sorted_by_fitness(record))
        ax = figure.add_subplot(grid[row, col])
        im = ax.imshow(matrix, cmap=DECISION_CMAP, interpolation="nearest", vmin=0, vmax=N_DECISION_CATEGORIES - 1, aspect="auto")
        ax.set_title(f"Generation {tracked_generations[idx]}", fontsize=fs_title)
        ax.set_xlabel("Run index", fontsize=fs_axis)
        ax.set_ylabel("Network (best→worst)", fontsize=fs_axis)
        ax.tick_params(labelsize=fs_axis - 1)
        if im_ref is None:
            im_ref = im
        csv_rows.extend(_decision_rows(tracked_generations[idx], matrix))

    # hide unused slots in the last row
    for spare in range(n_plots, n_rows * n_cols):
        row, col = divmod(spare, n_cols)
        figure.add_subplot(grid[row, col]).set_visible(False)

    colorbar_axis = figure.add_subplot(grid[n_rows, :])
    colorbar = figure.colorbar(im_ref, cax=colorbar_axis, orientation="horizontal", ticks=np.arange(0, N_DECISION_CATEGORIES, 1))
    colorbar.ax.set_xticklabels(DECISION_LABELS, fontsize=fs_colorbar)

    figure.suptitle("Decisions: all tracked generations (sorted by fitness)", fontsize=fs_title + 2, y=1.0)
    figure.savefig(_prefixed_path(plot_dir, ALL_DECISIONS_FILENAME), bbox_inches="tight")
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, ALL_DECISIONS_CSV_FILENAME), csv_rows)


def _save_event_counts_plot(plot_dir, tracked_records, tracked_generations, colors):
    """Save a grid of grouped bar plots, one row per training-phase segment in
    TRAINING_PHASE_LAYOUT (replay segments skipped). Each row is a group per tracked
    generation, with one bar per event (x, Lx, Rx, lx, rx, L, R, l, r), showing what %
    of that generation's events - within that training-phase segment's runs only -
    each event type accounted for."""
    event_labels = DECISION_LABELS[1:]  # exclude "." (not a real event, just padding)
    event_colors = DECISION_COLORS[1:]  # same event -> color mapping as the decision heatmaps
    n_events = len(event_labels)
    n_gens = len(tracked_records)
    n_rows = len(TRAINING_PHASE_LAYOUT)

    # cumulative run-index boundaries for each phase segment, matching the order
    # fitness.py's _concat_tracking_segments concatenated them in
    run_start = 0
    phase_run_ranges = []
    for phase_type, num_runs in TRAINING_PHASE_LAYOUT:
        phase_run_ranges.append((phase_type, run_start, run_start + num_runs))
        run_start += num_runs

    fig_w = max(12.0, n_events * n_gens * 0.35)
    figure, axes = plt.subplots(nrows=n_rows, ncols=1, figsize=(fig_w, 6 * n_rows), dpi=PLOT_DPI, squeeze=False)

    group_width = 0.8
    bar_width = group_width / n_events
    x_base = np.arange(n_gens)
    csv_rows = []

    for row_idx, (phase_type, seg_start, seg_end) in enumerate(phase_run_ranges):
        axis = axes[row_idx, 0]
        counts = np.zeros((n_gens, n_events), dtype=int)
        for g_idx, record in enumerate(tracked_records):
            matrix = _record_decision_matrix(record)[:, seg_start:seg_end]  # counts don't depend on fitness sort order
            for e_idx in range(n_events):
                counts[g_idx, e_idx] = int((matrix == e_idx + 1).sum())

        # percentage of that generation's events (the 9 real event types only - "." padding
        # is excluded from both the numerator and the denominator), so each generation's bars
        # sum to 100% regardless of how many runs actually completed.
        totals = counts.sum(axis=1, keepdims=True)
        percentages = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0) * 100.0

        for g_idx in range(n_gens):
            for e_idx in range(n_events):
                csv_rows.append({
                    "phase_type": phase_type,
                    "run_range_start": seg_start,
                    "run_range_end": seg_end - 1,
                    "generation": tracked_generations[g_idx],
                    "event_label": event_labels[e_idx],
                    "count": int(counts[g_idx, e_idx]),
                    "percentage_of_generation_events": percentages[g_idx, e_idx],
                })

        for e_idx in range(n_events):
            offset = (e_idx - (n_events - 1) / 2) * bar_width
            axis.bar(
                x_base + offset, percentages[:, e_idx], width=bar_width,
                color=event_colors[e_idx], edgecolor="#333333", linewidth=0.3,
                label=event_labels[e_idx] if row_idx == 0 else None,
            )

        axis.set_xticks(x_base)
        axis.set_xticklabels([f"gen {g}" for g in tracked_generations])
        axis.set_xlabel("Generation")
        axis.set_ylabel("% of events in generation")
        axis.set_title(f"{phase_type} (runs {seg_start}-{seg_end - 1})")
        axis.grid(True, axis="y", alpha=0.2)

    axes[0, 0].legend(ncol=min(n_events, 9), fontsize=8)
    figure.suptitle("Event distribution (%) by training phase across tracked generations")
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, EVENT_COUNTS_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, EVENT_COUNTS_CSV_FILENAME), csv_rows)


def _save_reward_hist_plot(plot_dir, tracked_records, tracked_generations, colors):
    """Save overlapping line histograms for tracked-generation fitness distributions."""
    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=PLOT_DPI)

    all_values = [record["fitness"].numpy() for record in tracked_records]
    global_min = min(values.min() for values in all_values)
    global_max = max(values.max() for values in all_values)
    start = HIST_BIN_WIDTH * np.floor(global_min / HIST_BIN_WIDTH)
    end = HIST_BIN_WIDTH * np.ceil(global_max / HIST_BIN_WIDTH)
    bin_edges = np.arange(start, end + HIST_BIN_WIDTH, HIST_BIN_WIDTH)
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    csv_rows = []
    for idx, record in enumerate(tracked_records):
        counts, _ = np.histogram(record["fitness"].numpy(), bins=bin_edges, density=True)
        axis.plot(x_centers, counts, color=colors[idx], linewidth=2.0, label=f"gen {tracked_generations[idx]}")
        for bin_center, density in zip(x_centers, counts):
            csv_rows.append({
                "generation": tracked_generations[idx],
                "fitness_bin_center": bin_center,
                "density": density,
            })

    axis.set_title("Fitness distribution across tracked generations")
    axis.set_xlabel("Fitness")
    axis.set_ylabel("Distribution density")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, REWARD_HIST_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, REWARD_HIST_CSV_FILENAME), csv_rows)


def _plot_frob_panel(axis, tracked_records, tracked_generations, colors, key, panel_title, stage_label):
    """Draw overlapping Frobenius traces for one panel. Returns this panel's CSV rows
    (stage_label distinguishes "start" vs "end" once combined with the other panel)."""
    csv_rows = []
    for idx, record in enumerate(tracked_records):
        values = np.sort(record[key].numpy())
        x = np.arange(values.shape[0])

        axis.plot(x, values, color=colors[idx], linewidth=1.2, alpha=0.9, label=f"gen {tracked_generations[idx]}")
        for sorted_index, value in zip(x, values):
            csv_rows.append({
                "generation": tracked_generations[idx],
                "stage": stage_label,
                "sorted_index": int(sorted_index),
                "frobenius_norm": value,
            })

    axis.set_title(panel_title)
    axis.set_xlabel("Network index (sorted)")
    axis.set_ylabel("Frobenius norm")
    axis.grid(True, alpha=0.2)
    return csv_rows


def _save_frobenius_plot(plot_dir, tracked_records, tracked_generations, colors):
    """Save side-by-side Frobenius plots for start and end weight norms."""
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), dpi=PLOT_DPI)
    csv_rows = _plot_frob_panel(axes[0], tracked_records, tracked_generations, colors, "frob_start", "Starting weights", "start")
    csv_rows += _plot_frob_panel(axes[1], tracked_records, tracked_generations, colors, "frob_end", "End-of-eval weights", "end")
    axes[1].legend()
    figure.suptitle("Frobenius norm evolution across tracked generations")
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, FROBENIUS_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, FROBENIUS_CSV_FILENAME), csv_rows)


def _plot_weight_distribution_panel(axis, tracked_records, tracked_generations, colors, key, panel_title, stage_label):
    """Draw overlapping line histograms for raw weight values. Returns this panel's CSV rows."""
    all_values = [record[key].numpy().reshape(-1) for record in tracked_records]
    global_min = min(values.min() for values in all_values)
    global_max = max(values.max() for values in all_values)
    if global_min == global_max:
        global_min -= 0.5
        global_max += 0.5

    bin_edges = np.linspace(global_min, global_max, WEIGHT_HIST_BINS + 1)
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    csv_rows = []
    for idx, values in enumerate(all_values):
        counts, _ = np.histogram(values, bins=bin_edges, density=True)
        axis.plot(x_centers, counts, color=colors[idx], linewidth=1.6, alpha=0.9, label=f"gen {tracked_generations[idx]}")
        for bin_center, density in zip(x_centers, counts):
            csv_rows.append({
                "generation": tracked_generations[idx],
                "stage": stage_label,
                "weight_bin_center": bin_center,
                "density": density,
            })

    axis.set_title(panel_title)
    axis.set_xlabel("Weight value")
    axis.set_ylabel("Distribution density")
    axis.grid(True, alpha=0.2)
    return csv_rows


def _save_weight_distribution_plot(plot_dir, tracked_records, tracked_generations, colors):
    """Save side-by-side line histograms for start and end weight distributions."""
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), dpi=PLOT_DPI)
    csv_rows = _plot_weight_distribution_panel(
        axes[0], tracked_records, tracked_generations, colors, "weights_start", "Starting weights", "start"
    )
    csv_rows += _plot_weight_distribution_panel(
        axes[1], tracked_records, tracked_generations, colors, "weights_end", "End-of-eval weights", "end"
    )
    axes[1].legend()
    figure.suptitle("Weight-value distributions across tracked generations")
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, WEIGHT_DISTRIBUTION_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, WEIGHT_DISTRIBUTION_CSV_FILENAME), csv_rows)


def _save_reward_evolution_plot(plot_dir, reward_evolution):
    """Save all-generation line plot for mean, median, and best population fitness."""
    generations = np.array(reward_evolution["generation"])
    mean_eval = np.array(reward_evolution["mean_eval"])
    median_eval = np.array(reward_evolution["median_eval"])
    pop_best_eval = np.array(reward_evolution["pop_best_eval"])

    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=PLOT_DPI)
    axis.plot(generations, mean_eval, color=REWARD_EVOLUTION_COLORS[0], linewidth=2.0, label="mean")
    axis.plot(generations, median_eval, color=REWARD_EVOLUTION_COLORS[1], linewidth=2.0, label="median")
    axis.plot(generations, pop_best_eval, color=REWARD_EVOLUTION_COLORS[2], linewidth=2.0, label="best")
    axis.set_title("Fitness evolution across all generations")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Fitness")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, REWARD_EVOLUTION_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, REWARD_EVOLUTION_CSV_FILENAME), _columns_to_rows({
        "generation": generations, "mean_eval": mean_eval, "median_eval": median_eval, "pop_best_eval": pop_best_eval,
    }))


def _save_training_reward_evolution_plot(plot_dir, reward_evolution):
    """Save all-generation line plot for mean, median, and best training reward (summed across all paradigm training phases, pre-L1)."""
    generations = np.array(reward_evolution["generation"])
    mean_tr = np.array(reward_evolution["training_reward_mean"])
    median_tr = np.array(reward_evolution["training_reward_median"])
    best_tr = np.array(reward_evolution["training_reward_best"])

    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=PLOT_DPI)
    axis.plot(generations, mean_tr, color=REWARD_EVOLUTION_COLORS[0], linewidth=2.0, label="mean")
    axis.plot(generations, median_tr, color=REWARD_EVOLUTION_COLORS[1], linewidth=2.0, label="median")
    axis.plot(generations, best_tr, color=REWARD_EVOLUTION_COLORS[2], linewidth=2.0, label="best")
    axis.set_title("Training reward evolution across all generations (summed across paradigm, pre-L1)")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Training reward")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, TRAINING_REWARD_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, TRAINING_REWARD_CSV_FILENAME), _columns_to_rows({
        "generation": generations, "mean_training_reward": mean_tr, "median_training_reward": median_tr, "best_training_reward": best_tr,
    }))


def _save_l1_evolution_plot(plot_dir, reward_evolution):
    """Save all-generation line plot for mean, median, and max L1 penalty."""
    generations = np.array(reward_evolution["generation"])
    mean_l1 = np.array(reward_evolution["l1_penalty_mean"])
    median_l1 = np.array(reward_evolution["l1_penalty_median"])
    max_l1 = np.array(reward_evolution["l1_penalty_best"])

    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=PLOT_DPI)
    axis.plot(generations, mean_l1, color=REWARD_EVOLUTION_COLORS[0], linewidth=2.0, label="mean")
    axis.plot(generations, median_l1, color=REWARD_EVOLUTION_COLORS[1], linewidth=2.0, label="median")
    axis.plot(generations, max_l1, color=REWARD_EVOLUTION_COLORS[2], linewidth=2.0, label="max")
    axis.set_title("L1 penalty evolution across all generations")
    axis.set_xlabel("Generation")
    axis.set_ylabel("L1 penalty")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, L1_EVOLUTION_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, L1_EVOLUTION_CSV_FILENAME), _columns_to_rows({
        "generation": generations, "mean_l1_penalty": mean_l1, "median_l1_penalty": median_l1, "max_l1_penalty": max_l1,
    }))


def _save_input_weighing_plot(plot_dir, cue_importance_history):
    """Save tracked-generation line plot of ablation-based cue importance: how much
    unregularized reward is lost when the context cue (resp. sensory cue, resp. the
    online reward-input signal) is clipped to zero, relative to the actual evaluation
    condition. Near zero means the network isn't using that cue at all; a large drop
    means it depends on it heavily."""
    generations = np.array(cue_importance_history["generation"])
    context_mean = np.array(cue_importance_history["context_importance_mean"])
    context_min = np.array(cue_importance_history["context_importance_min"])
    context_max = np.array(cue_importance_history["context_importance_max"])
    sensory_mean = np.array(cue_importance_history["sensory_importance_mean"])
    sensory_min = np.array(cue_importance_history["sensory_importance_min"])
    sensory_max = np.array(cue_importance_history["sensory_importance_max"])
    reward_mean = np.array(cue_importance_history["reward_importance_mean"])
    reward_min = np.array(cue_importance_history["reward_importance_min"])
    reward_max = np.array(cue_importance_history["reward_importance_max"])

    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=PLOT_DPI)
    axis.plot(generations, context_mean, color=REWARD_EVOLUTION_COLORS[0], linewidth=2.0, label="context cue")
    axis.fill_between(generations, context_min, context_max, color=REWARD_EVOLUTION_COLORS[0], alpha=0.15)
    axis.plot(generations, sensory_mean, color=REWARD_EVOLUTION_COLORS[2], linewidth=2.0, label="sensory cue")
    axis.fill_between(generations, sensory_min, sensory_max, color=REWARD_EVOLUTION_COLORS[2], alpha=0.15)
    axis.plot(generations, reward_mean, color=REWARD_EVOLUTION_COLORS[1], linewidth=2.0, label="reward signal")
    axis.fill_between(generations, reward_min, reward_max, color=REWARD_EVOLUTION_COLORS[1], alpha=0.15)
    axis.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    axis.set_title("Input cue importance across tracked generations (reward lost when cue is ablated)")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Reward lost when cue is clipped to zero")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, INPUT_WEIGHING_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, INPUT_WEIGHING_CSV_FILENAME), _columns_to_rows({
        "generation": generations,
        "context_importance_mean": context_mean, "context_importance_min": context_min, "context_importance_max": context_max,
        "sensory_importance_mean": sensory_mean, "sensory_importance_min": sensory_min, "sensory_importance_max": sensory_max,
        "reward_importance_mean": reward_mean, "reward_importance_min": reward_min, "reward_importance_max": reward_max,
    }))


def _save_sensory_cue_plot(plot_dir, tracked_records):
    """Save side-by-side bar charts of sensory cue distribution for first and last tracked generation."""
    first_record = tracked_records[0]
    last_record = tracked_records[-1]

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=PLOT_DPI, sharey=True)

    csv_rows = []
    for axis, record, title_suffix in (
        (axes[0], first_record, f"Generation {first_record['generation']}"),
        (axes[1], last_record, f"Generation {last_record['generation']}"),
    ):
        cues = record["sensory_cue_by_run"].numpy()   # [pop, num_runs]
        num_runs = cues.shape[1]
        num_cue_types = int(cues.max().item()) + 1
        cue_labels = [f"cue_{chr(65 + i)}" for i in range(num_cue_types)]

        counts = np.array([(cues == i).sum() for i in range(num_cue_types)], dtype=float)
        percentages = counts / num_runs / cues.shape[0] * 100.0

        bars = axis.bar(cue_labels, percentages, color=PALETTE_COLORS[:num_cue_types])
        axis.set_title(title_suffix)
        axis.set_xlabel("Sensory cue")
        axis.set_ylim(0, 100)
        for bar, pct in zip(bars, percentages):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for cue_label, count, pct in zip(cue_labels, counts, percentages):
            csv_rows.append({
                "generation": record["generation"],
                "cue_label": cue_label,
                "count": int(count),
                "percentage_of_runs": pct,
            })

    axes[0].set_ylabel("Percentage of runs (%)")
    figure.suptitle("Sensory cue distribution across maze runs")
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, SENSORY_CUE_FILENAME))
    plt.close(figure)

    write_csv(_prefixed_path(plot_dir, SENSORY_CUE_CSV_FILENAME), csv_rows)


def _save_transfer_metrics_plot(plot_dir, transfer_metrics_history):
    """Save tracked-generation FWT/BWT plot - Lopez-Paz & Ranzato (2017) "Gradient
    Episodic Memory for Continual Learning", Eqs. 3-4, generalized to however many
    distinct training tasks THIS paradigm actually has (see sim_core/fitness.py's
    _measure_transfer_metrics; TASK_ORDER_LABELS, module-level, reflects this run's
    actual paradigm order/count - e.g. ["A","B"] or, once a third task exists,
    ["A","B","C"] - so titles/columns are correct regardless of task count or order).
    Both metrics are already averaged over however many task-pair terms T-1 produces
    (T = len(TASK_ORDER_LABELS)), matching the paper's single reported FWT/BWT number
    per model - so the plot itself never grows with T, only the CSV's traceability
    columns do (one _mean column per underlying R_i,i / R_T,i / R_i-1,i / b_i probe -
    see transfer_metrics_history's dynamically-named keys)."""
    generations = np.array(transfer_metrics_history["generation"])
    fwt_mean = np.array(transfer_metrics_history["fwt_mean"])
    fwt_min = np.array(transfer_metrics_history["fwt_min"])
    fwt_max = np.array(transfer_metrics_history["fwt_max"])
    bwt_mean = np.array(transfer_metrics_history["bwt_mean"])
    bwt_min = np.array(transfer_metrics_history["bwt_min"])
    bwt_max = np.array(transfer_metrics_history["bwt_max"])
    task_chain = " -> ".join(TASK_ORDER_LABELS)

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), dpi=PLOT_DPI)
    axes[0].plot(generations, fwt_mean, color=REWARD_EVOLUTION_COLORS[0], linewidth=2.0)
    axes[0].fill_between(generations, fwt_min, fwt_max, color=REWARD_EVOLUTION_COLORS[0], alpha=0.15)
    axes[0].axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    axes[0].set_title(f"Forward transfer (mean over consecutive pairs): {task_chain}")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("FWT (reward/run)")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(generations, bwt_mean, color=REWARD_EVOLUTION_COLORS[1], linewidth=2.0)
    axes[1].fill_between(generations, bwt_min, bwt_max, color=REWARD_EVOLUTION_COLORS[1], alpha=0.15)
    axes[1].axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    axes[1].set_title(f"Backward transfer (mean over first {len(TASK_ORDER_LABELS) - 1} task(s)): {task_chain}, after all trained")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("BWT (reward/run)")
    axes[1].grid(True, alpha=0.2)

    figure.suptitle("Forward/backward knowledge transfer (Lopez-Paz & Ranzato 2017) across tracked generations")
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, TRANSFER_METRICS_FILENAME))
    plt.close(figure)

    n_gens = len(generations)
    base_keys = ("generation", "fwt_mean", "fwt_min", "fwt_max", "bwt_mean", "bwt_min", "bwt_max")
    component_keys = sorted(key for key in transfer_metrics_history if key not in base_keys)
    csv_columns = {"generation": generations, "task_order": [task_chain] * n_gens}
    for key in base_keys[1:]:
        csv_columns[key] = np.array(transfer_metrics_history[key])
    for key in component_keys:
        csv_columns[key] = np.array(transfer_metrics_history[key])
    write_csv(_prefixed_path(plot_dir, TRANSFER_METRICS_CSV_FILENAME), _columns_to_rows(csv_columns))


def _save_all_plots_and_results(searcher, history, pgpe_history):
    """Create RUN_DIR, copy the config file used for this run into it, save all
    tracking plots, and write the full numeric results (final genomes + history)
    to RESULTS_FILENAME - all into the one timestamped, collision-proof folder."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, RUN_DIR / CONFIG_PATH.name)

    tracked_generations = history["tracked_generations"]
    tracked_records = history["tracked_records"]
    reward_evolution = history["reward_evolution"]
    cue_importance_history = history["cue_importance_history"]
    transfer_metrics_history = history["transfer_metrics_history"]
    colors = _generation_colors(tracked_generations)

    _save_decisions_plot(RUN_DIR, tracked_records)
    _save_all_decisions_plot(RUN_DIR, tracked_records, tracked_generations)
    _save_event_counts_plot(RUN_DIR, tracked_records, tracked_generations, colors)
    _save_reward_hist_plot(RUN_DIR, tracked_records, tracked_generations, colors)
    _save_frobenius_plot(RUN_DIR, tracked_records, tracked_generations, colors)
    _save_weight_distribution_plot(RUN_DIR, tracked_records, tracked_generations, colors)
    _save_reward_evolution_plot(RUN_DIR, reward_evolution)
    _save_training_reward_evolution_plot(RUN_DIR, reward_evolution)
    _save_l1_evolution_plot(RUN_DIR, reward_evolution)
    _save_input_weighing_plot(RUN_DIR, cue_importance_history)
    _save_transfer_metrics_plot(RUN_DIR, transfer_metrics_history)
    _save_sensory_cue_plot(RUN_DIR, tracked_records)
    _save_pgpe_params_plot(RUN_DIR, pgpe_history)
    _save_pgpe_fitness_plot(RUN_DIR, pgpe_history)
    _save_pgpe_stagnation_plot(RUN_DIR, pgpe_history)

    run_metadata = {
        "run_name": RUN_NAME,
        "timestamp": RUN_TIMESTAMP,
        "config_filename": CONFIG_PATH.name,
        "torch_version": torch.__version__,
        "evotorch_version": evotorch.__version__,
    }
    save_results_h5(RUN_DIR / results_filename(RUN_NAME), CONFIG, run_metadata, searcher, history, pgpe_history)

    print(f"\nSaved plots + results to: {RUN_DIR}")


# ==== 5) EVOLUTION RUN ==========================================================
_validate_device_or_raise(DEVICE)

torch.manual_seed(MASTER_SEED)
torch.cuda.manual_seed_all(MASTER_SEED)
torch.use_deterministic_algorithms(True)

noise_generator = torch.Generator(device=DEVICE)
noise_generator.manual_seed(NOISE_SEED)
reward_generator = torch.Generator(device=DEVICE)
reward_generator.manual_seed(REWARD_SEED)
test_generator = torch.Generator(device=DEVICE)
test_generator.manual_seed(TEST_SEED)
weight_init_generator = torch.Generator(device=DEVICE)
weight_init_generator.manual_seed(WEIGHT_INIT_SEED)

center_init = torch.zeros(GENOME_LENGTH, device=DEVICE)

objective = functools.partial(
    fitness_function,
    device=DEVICE,
    noise_generator=noise_generator,
    reward_generator=reward_generator,
    test_generator=test_generator,
    weight_init_generator=weight_init_generator,
    l1_lambda=L1_LAMBDA,
    context_cues_on=EVO_CONTEXT_CUES_ON,
    sensory_cues_on=EVO_SENSORY_CUES_ON,
    evo_plasticity_on=EVO_PLASTICITY_ON,
    paradigm_phases=PARADIGM_PHASES,
)

configure_printing(
    total_generations=NUM_GENERATIONS,
    print_interval=TRACKED_PER_INTERVAL,
    max_networks_preview=MAX_NETWORKS_PREVIEW,
    max_runs_preview=MAX_RUNS_PREVIEW,
    chain_label=CHAIN_LABEL,
    config_name=RUN_NAME,
    early_termination_enabled=EARLY_TERMINATION_ENABLED,
)

problem = Problem(
    objective_sense="max",
    objective_func=objective,
    solution_length=GENOME_LENGTH,
    device=DEVICE,
    vectorized=True,
)

searcher = PGPE(
    problem,
    popsize=SEARCH_POPSIZE,
    center_learning_rate=CENTER_LEARNING_RATE,
    stdev_learning_rate=STDEV_LEARNING_RATE,
    radius_init=RADIUS_INIT,
    center_init=center_init,
    optimizer="clipup",
    optimizer_config={"max_speed": MAX_SPEED, "momentum": MOMENTUM},
    stdev_min=PGPE_STDEV_MIN,  # None -> no floor, identical to this file's old behavior
)

perturb_generator = None
if PGPE_CENTER_PERTURB_ENABLED:
    perturb_generator = torch.Generator(device=DEVICE)
    perturb_generator.manual_seed(PGPE_PERTURB_SEED)

pgpe_history = _init_pgpe_history()
best_fitness_ever = float("-inf")
generations_since_improvement = 0
for generation_idx in range(1, NUM_GENERATIONS + 1):
    searcher.step()
    history_snapshot = get_printing_history()
    reward_evolution = history_snapshot["reward_evolution"]

    restart_triggered = False
    if PGPE_RESTART_ENABLED:
        current_best = float(reward_evolution["pop_best_eval"][-1])
        if current_best > (best_fitness_ever + PGPE_RESTART_MIN_IMPROVEMENT):
            best_fitness_ever = current_best
            generations_since_improvement = 0
        else:
            generations_since_improvement += 1

        if generations_since_improvement >= PGPE_RESTART_PATIENCE:
            restart_triggered = True
            _restart_pgpe_distribution(searcher)
            generations_since_improvement = 0

    center_perturbed = False
    if PGPE_CENTER_PERTURB_ENABLED and (generation_idx % PGPE_CENTER_PERTURB_INTERVAL == 0):
        center_perturbed = True
        _perturb_pgpe_center(searcher, perturb_generator)

    _collect_pgpe_history(
        searcher,
        reward_evolution,
        pgpe_history,
        generations_since_improvement=generations_since_improvement,
        restart_triggered=restart_triggered,
        center_perturbed=center_perturbed,
    )
    if should_terminate_early():
        break  # population converged to always turning one direction - see fitness.py's
               # _print_tracking_block; everything below saves normally, just with fewer
               # generations than NUM_GENERATIONS actually happened

history = get_printing_history()
_save_all_plots_and_results(searcher, history, pgpe_history)