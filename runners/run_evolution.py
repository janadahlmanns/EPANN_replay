"""Run PGPE with tracking output and save end-of-run plots + full numeric results."""

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
matplotlib.use("Agg")  # headless -- this only ever calls savefig(), never plt.show(), and a
                        # remote GPU server reached over VPN may have no display/Tk at all
import matplotlib.pyplot as plt
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import PGPE

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
from analysis.results_io import results_filename, save_results_h5
from sim_core import constants, genome_codec
from sim_core.constants import INPUT_SENSORY_A, INPUT_SENSORY_B
from sim_core.fitness import configure_printing, fitness_function, get_printing_history
from sim_core.paradigm import PHASE_REPLAY, parse_paradigm

# ==== 2) CONFIG LOADING + OUTPUT LOCATION =======================================
# Results always live under the hardcoded DATA_ROOT -- not a user choice. The
# config is found by joining CONFIGS_ROOT with the given name + ".json" -- that's
# it, nothing else: for a plain name that's configs/<name>.json; run_batch.py
# reaches configs/batch_to_run/<name>.json the same way, by passing
# "batch_to_run/<name>" as that same argument. No defaults/fallbacks on the
# config contents -- a missing or malformed field fails loudly (KeyError), on purpose.
CONFIGS_ROOT = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"

if len(sys.argv) != 5:
    raise ValueError("Usage: python run_evolution.py <config_name> <experiment_name> <device> <chain_label>")
CONFIG_PATH = CONFIGS_ROOT / f"{sys.argv[1]}.json"
OUTPUT_ROOT = DATA_ROOT / sys.argv[2]
CHAIN_LABEL = sys.argv[4]  # printed as "CHAIN <label>" in terminal output -- run_batch.py passes
                            # its chain index; a manual run can pass anything, e.g. "manual"
with open(CONFIG_PATH, "r", encoding="utf-8") as _config_file:
    CONFIG = json.load(_config_file)

RUN_NAME = CONFIG_PATH.stem  # output folder is always named after the input config file
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # year->...->second, so
                                                                     # alphabetical (file explorer)
                                                                     # order is chronological order
RUN_DIR = OUTPUT_ROOT / f"{RUN_NAME}_{RUN_TIMESTAMP}"  # everything this run produces lives here
DEVICE = sys.argv[3]  # required CLI input -- always wins, even if the config json still has its
                       # own (by-now-vestigial) "device" field
N_NEURONS = CONFIG["n_neurons"]  # total neurons; see constants.configure_network() for the fixed
                                  # input/output assignment + derived hidden-neuron count
MASTER_SEED = CONFIG["master_seed"]
NOISE_SEED = CONFIG["noise_seed"]
REWARD_SEED = CONFIG["reward_seed"]
TEST_SEED = CONFIG["test_seed"]  # dedicated RNG stream for cue-importance measurement only -- never
                                  # touches noise_generator/reward_generator, so the tracking interval
                                  # can't change the run
WEIGHT_INIT_SEED = CONFIG["weight_init_seed"]  # dedicated RNG stream for the fresh per-lifetime initial-weight
                                                # draw (sample_initial_weights) -- never touches any other stream

EVO_CONTEXT_CUES_ON = CONFIG["evo_context_cues_on"]  # if False, context-cue input neurons are clipped to zero during evolution
EVO_SENSORY_CUES_ON = CONFIG["evo_sensory_cues_on"]  # if False, sensory-cue input neurons are clipped to zero during evolution
EVO_PLASTICITY_ON = CONFIG["evo_plasticity_on"]  # if False, eta is forced to all zeros regardless of genome -> no plasticity, frozen weights

# Per-evaluation phase sequence: comma-separated (phase, value) pairs, where phase
# is one of "trainA"/"trainB" (value = number of maze runs) or "replay" (value =
# number of ticks). Parsed eagerly below so a malformed string fails at import time.
PARADIGM = CONFIG["paradigm"]
PARADIGM_PHASES = parse_paradigm(PARADIGM)
# ordered list of (phase_type, num_runs) for training phases only, replay skipped --
# this must stay in the same order fitness.py concatenates tracking segments in
TRAINING_PHASE_LAYOUT = [(phase_type, value) for phase_type, value in PARADIGM_PHASES if phase_type != PHASE_REPLAY]

NUM_GENERATIONS = CONFIG["num_generations"]
SEARCH_POPSIZE = CONFIG["search_popsize"]
RADIUS_INIT = CONFIG["radius_init"]     # radius of the initial search hypersphere in genome space (GENOME_LENGTH-dim), sweep/ optimize
MAX_SPEED = RADIUS_INIT / 15.0          # evotorch's rule of thumb from the ClipUp paper: max_speed = radius / 15.0, adjust the 15.0 to optimize
CENTER_LEARNING_RATE = MAX_SPEED / 2    # this is the step size in the ClipUp paper
STDEV_LEARNING_RATE = CONFIG["stdev_learning_rate"]
MOMENTUM = CONFIG["momentum"]

L1_LAMBDA = CONFIG["l1_lambda"]

TRACKED_PER_INTERVAL = CONFIG["tracked_per_interval"]
MAX_NETWORKS_PREVIEW = CONFIG["max_networks_preview"]
MAX_RUNS_PREVIEW = CONFIG["max_runs_preview"]
HIST_BIN_WIDTH = CONFIG["hist_bin_width"]
PLOT_DPI = 180  # presentation-only, not an experiment parameter -- stays fixed

# Network layout + tunable sim_core constants (reward shaping + CTRNN dynamics) --
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
PGPE_PARAMS_FILENAME = "pgpe_params.png"
PGPE_FITNESS_FILENAME = "pgpe_fitness.png"
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
    }


def _collect_pgpe_history(searcher, reward_evolution, pgpe_history):
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


# ==== 4) PLOTTING HELPERS ======================================================
def _prefixed_path(plot_dir, filename):
    """Prefix every saved figure's filename with RUN_NAME, e.g. 'decisions.png' -> 'NAME_decisions.png'."""
    return plot_dir / f"{RUN_NAME}_{filename}"


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


def _save_event_counts_plot(plot_dir, tracked_records, tracked_generations, colors):
    """Save a grid of grouped bar plots, one row per training-phase segment in
    TRAINING_PHASE_LAYOUT (replay segments skipped). Each row is a group per tracked
    generation, with one bar per event (x, Lx, Rx, lx, rx, L, R, l, r), showing what %
    of that generation's events -- within that training-phase segment's runs only --
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

    for row_idx, (phase_type, seg_start, seg_end) in enumerate(phase_run_ranges):
        axis = axes[row_idx, 0]
        counts = np.zeros((n_gens, n_events), dtype=int)
        for g_idx, record in enumerate(tracked_records):
            matrix = _record_decision_matrix(record)[:, seg_start:seg_end]  # counts don't depend on fitness sort order
            for e_idx in range(n_events):
                counts[g_idx, e_idx] = int((matrix == e_idx + 1).sum())

        # percentage of that generation's events (the 9 real event types only -- "." padding
        # is excluded from both the numerator and the denominator), so each generation's bars
        # sum to 100% regardless of how many runs actually completed.
        totals = counts.sum(axis=1, keepdims=True)
        percentages = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0) * 100.0

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

    for idx, record in enumerate(tracked_records):
        counts, _ = np.histogram(record["fitness"].numpy(), bins=bin_edges, density=True)
        axis.plot(x_centers, counts, color=colors[idx], linewidth=2.0, label=f"gen {tracked_generations[idx]}")

    axis.set_title("Fitness distribution across tracked generations")
    axis.set_xlabel("Fitness")
    axis.set_ylabel("Distribution density")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, REWARD_HIST_FILENAME))
    plt.close(figure)


def _plot_frob_panel(axis, tracked_records, tracked_generations, colors, key, panel_title):
    """Draw overlapping Frobenius traces for one panel."""
    for idx, record in enumerate(tracked_records):
        values = np.sort(record[key].numpy())
        x = np.arange(values.shape[0])

        axis.plot(x, values, color=colors[idx], linewidth=1.2, alpha=0.9, label=f"gen {tracked_generations[idx]}")

    axis.set_title(panel_title)
    axis.set_xlabel("Network index (sorted)")
    axis.set_ylabel("Frobenius norm")
    axis.grid(True, alpha=0.2)


def _save_frobenius_plot(plot_dir, tracked_records, tracked_generations, colors):
    """Save side-by-side Frobenius plots for start and end weight norms."""
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), dpi=PLOT_DPI)
    _plot_frob_panel(axes[0], tracked_records, tracked_generations, colors, "frob_start", "Starting weights")
    _plot_frob_panel(axes[1], tracked_records, tracked_generations, colors, "frob_end", "End-of-eval weights")
    axes[1].legend()
    figure.suptitle("Frobenius norm evolution across tracked generations")
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, FROBENIUS_FILENAME))
    plt.close(figure)


def _plot_weight_distribution_panel(axis, tracked_records, tracked_generations, colors, key, panel_title):
    """Draw overlapping line histograms for raw weight values."""
    all_values = [record[key].numpy().reshape(-1) for record in tracked_records]
    global_min = min(values.min() for values in all_values)
    global_max = max(values.max() for values in all_values)
    if global_min == global_max:
        global_min -= 0.5
        global_max += 0.5

    bin_edges = np.linspace(global_min, global_max, WEIGHT_HIST_BINS + 1)
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for idx, values in enumerate(all_values):
        counts, _ = np.histogram(values, bins=bin_edges, density=True)
        axis.plot(x_centers, counts, color=colors[idx], linewidth=1.6, alpha=0.9, label=f"gen {tracked_generations[idx]}")

    axis.set_title(panel_title)
    axis.set_xlabel("Weight value")
    axis.set_ylabel("Distribution density")
    axis.grid(True, alpha=0.2)


def _save_weight_distribution_plot(plot_dir, tracked_records, tracked_generations, colors):
    """Save side-by-side line histograms for start and end weight distributions."""
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), dpi=PLOT_DPI)
    _plot_weight_distribution_panel(
        axes[0], tracked_records, tracked_generations, colors, "weights_start", "Starting weights"
    )
    _plot_weight_distribution_panel(
        axes[1], tracked_records, tracked_generations, colors, "weights_end", "End-of-eval weights"
    )
    axes[1].legend()
    figure.suptitle("Weight-value distributions across tracked generations")
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, WEIGHT_DISTRIBUTION_FILENAME))
    plt.close(figure)


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


def _save_input_weighing_plot(plot_dir, cue_importance_history):
    """Save tracked-generation line plot of ablation-based cue importance: how much
    unregularized reward is lost when the context cue (resp. sensory cue) is clipped
    to zero, relative to the actual evaluation condition. Near zero means the network
    isn't using that cue at all; a large drop means it depends on it heavily."""
    generations = np.array(cue_importance_history["generation"])
    context_mean = np.array(cue_importance_history["context_importance_mean"])
    context_min = np.array(cue_importance_history["context_importance_min"])
    context_max = np.array(cue_importance_history["context_importance_max"])
    sensory_mean = np.array(cue_importance_history["sensory_importance_mean"])
    sensory_min = np.array(cue_importance_history["sensory_importance_min"])
    sensory_max = np.array(cue_importance_history["sensory_importance_max"])

    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=PLOT_DPI)
    axis.plot(generations, context_mean, color=REWARD_EVOLUTION_COLORS[0], linewidth=2.0, label="context cue")
    axis.fill_between(generations, context_min, context_max, color=REWARD_EVOLUTION_COLORS[0], alpha=0.15)
    axis.plot(generations, sensory_mean, color=REWARD_EVOLUTION_COLORS[2], linewidth=2.0, label="sensory cue")
    axis.fill_between(generations, sensory_min, sensory_max, color=REWARD_EVOLUTION_COLORS[2], alpha=0.15)
    axis.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    axis.set_title("Input cue importance across tracked generations (reward lost when cue is ablated)")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Reward lost when cue is clipped to zero")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, INPUT_WEIGHING_FILENAME))
    plt.close(figure)


def _save_sensory_cue_plot(plot_dir, tracked_records):
    """Save side-by-side bar charts of sensory cue distribution for first and last tracked generation."""
    first_record = tracked_records[0]
    last_record = tracked_records[-1]

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=PLOT_DPI, sharey=True)

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

    axes[0].set_ylabel("Percentage of runs (%)")
    figure.suptitle("Sensory cue distribution across maze runs")
    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, SENSORY_CUE_FILENAME))
    plt.close(figure)


def _save_all_plots_and_results(searcher, history, pgpe_history):
    """Create RUN_DIR, copy the config file used for this run into it, save all
    tracking plots, and write the full numeric results (final genomes + history)
    to RESULTS_FILENAME -- all into the one timestamped, collision-proof folder."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, RUN_DIR / CONFIG_PATH.name)

    tracked_generations = history["tracked_generations"]
    tracked_records = history["tracked_records"]
    reward_evolution = history["reward_evolution"]
    cue_importance_history = history["cue_importance_history"]
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
    _save_sensory_cue_plot(RUN_DIR, tracked_records)
    _save_pgpe_params_plot(RUN_DIR, pgpe_history)
    _save_pgpe_fitness_plot(RUN_DIR, pgpe_history)

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
)

pgpe_history = _init_pgpe_history()
for _ in range(NUM_GENERATIONS):
    searcher.step()
    history_snapshot = get_printing_history()
    _collect_pgpe_history(searcher, history_snapshot["reward_evolution"], pgpe_history)

history = get_printing_history()
_save_all_plots_and_results(searcher, history, pgpe_history)