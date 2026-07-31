"""Run PGPE with tracking output and save end-of-run plots."""

# ==== 1) RNG DETERMINISM + PATH SETUP ==========================================
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger
from matplotlib.colors import ListedColormap

from sim_core.constants import INPUT_SENSORY_A, INPUT_SENSORY_B
from sim_core.fitness import configure_printing, fitness_function, get_printing_history
from sim_core.genome_codec import GENOME_LENGTH, GENOME_SPEC

# ==== 2) CONSTANTS / USER INPUTS ===============================================
RUN_NAME = "taskA"
DEVICE = "cuda"
MASTER_SEED = 0
NOISE_SEED = 1
REWARD_SEED = 2

NUM_GENERATIONS = 1000
SEARCH_POPSIZE = 200
RADIUS_INIT = 50            # radius of the initial search hypersphere in genome space (GENOME_LENGTH-dim), sweep/ optimize
MAX_SPEED = RADIUS_INIT / 15.0  # evotorch's rule of thumb from the ClipUp paper: max_speed = radius / 15.0, adjust the 15.0 to optimize
CENTER_LEARNING_RATE = MAX_SPEED / 2  # this is the step size in the ClipUp paper
STDEV_LEARNING_RATE = 0.1
MOMENTUM = 0.9

L1_LAMBDA = 1e-3

TRACKED_PER_INTERVAL = 40
MAX_NETWORKS_PREVIEW = 6
MAX_RUNS_PREVIEW = 20
HIST_BIN_WIDTH = 1
PLOT_DPI = 180

PLOTS_ROOT = Path("C:/EPANN_replay/data/plots")
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
DEBUG_PGPE_PARAMS_FILENAME = "debug_pgpe_params.png"
DEBUG_PGPE_FITNESS_FILENAME = "debug_pgpe_fitness.png"
REWARD_EVOLUTION_COLORS = ["#E07A5F", "#3D405B", "#81B29A"]
PALETTE_COLORS = ["#E07A5F", "#3D405B", "#81B29A", "#F2CC8F", "#F4F1DE"]
WEIGHT_HIST_BINS = 80

# Decision-outcome categories, encoded 0..9 in _decision_category_matrix.
# Color design: lightness encodes crash (light) vs. maze-end reward (dark);
# hue encodes turn direction (red=left, blue=right; gray/neutral = no turn);
# saturation encodes whether the chosen arm was correct (high) or not (low).
#   .  = no event                              -> white
#   x  = crash before/at the turn (no turn)     -> super light, near-white gray
#   Lx = correct left turn, then crash          -> light, highly saturated red
#   Rx = correct right turn, then crash         -> light, highly saturated blue
#   lx = wrong left turn, then crash            -> light, low saturation red
#   rx = wrong right turn, then crash           -> light, low saturation blue
#   L  = correct left turn, big reward at end   -> dark, highly saturated red
#   R  = correct right turn, big reward at end  -> dark, highly saturated blue
#   l  = wrong left turn, small reward at end   -> dark, low saturation red
#   r  = wrong right turn, small reward at end  -> dark, low saturation blue
DECISION_COLORS = [
    "#ffffff",  # .
    "#f0f0f0",  # x
    "#f49a9a",  # Lx
    "#9abff4",  # Rx
    "#dfc3c3",  # lx
    "#c3cfdf",  # rx
    "#9c1111",  # L
    "#114b9c",  # R
    "#7e4444",  # l
    "#445c7e",  # r
]
DECISION_LABELS = [".", "x", "Lx", "Rx", "lx", "rx", "L", "R", "l", "r"]
N_DECISION_CATEGORIES = len(DECISION_LABELS)


# ==== DEBUG: PGPE DIAGNOSTIC TRACKING (REMOVE AFTER TROUBLESHOOTING) ==========
_DEBUG_SENSORY_CUE_NEURON_INDICES = (INPUT_SENSORY_A, INPUT_SENSORY_B)


def _debug_sensory_cue_genome_indices():
    """Flat-genome indices touching sensory-cue neurons in any N-sized axis."""
    indices = []
    offset = 0
    for _, shape in GENOME_SPEC:
        for local_idx in np.ndindex(shape):
            if any(
                (axis_size > max(_DEBUG_SENSORY_CUE_NEURON_INDICES)) and (axis_value in _DEBUG_SENSORY_CUE_NEURON_INDICES)
                for axis_size, axis_value in zip(shape, local_idx)
            ):
                indices.append(offset + int(np.ravel_multi_index(local_idx, shape)))
        offset += int(np.prod(shape))
    return torch.tensor(indices, dtype=torch.long)


DEBUG_SENSORY_CUE_GENOME_INDICES = _debug_sensory_cue_genome_indices()


def _init_debug_pgpe_history():
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


def _debug_collect_pgpe_history(searcher, reward_evolution, debug_history):
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
            "DEBUG tracking could not find PGPE center/stdev in searcher.status or as searcher attributes."
        )

    center = center.detach().reshape(-1).float().cpu()
    stdev = stdev.detach().reshape(-1).float().cpu()
    if DEBUG_SENSORY_CUE_GENOME_INDICES.numel() == 0:
        raise RuntimeError("DEBUG sensory-cue index set is empty.")
    sensory_stdev = stdev[DEBUG_SENSORY_CUE_GENOME_INDICES]

    debug_history["generation"].append(generation)
    debug_history["center_norm"].append(float(torch.linalg.vector_norm(center).item()))
    debug_history["stdev_mean"].append(float(stdev.mean().item()))
    debug_history["stdev_min"].append(float(stdev.min().item()))
    debug_history["stdev_max"].append(float(stdev.max().item()))
    debug_history["stdev_sensory_mean"].append(float(sensory_stdev.mean().item()))
    debug_history["stdev_sensory_min"].append(float(sensory_stdev.min().item()))
    debug_history["stdev_sensory_max"].append(float(sensory_stdev.max().item()))
    debug_history["fitness_mean"].append(float(reward_evolution["mean_eval"][-1]))
    debug_history["fitness_max"].append(float(reward_evolution["pop_best_eval"][-1]))
    debug_history["fitness_std"].append(float(reward_evolution["std_eval"][-1]))


def _save_debug_pgpe_params_plot(plot_dir, debug_pgpe_history):
    generations = np.array(debug_pgpe_history["generation"])
    center_norm = np.array(debug_pgpe_history["center_norm"])
    stdev_mean = np.array(debug_pgpe_history["stdev_mean"])
    stdev_min = np.array(debug_pgpe_history["stdev_min"])
    stdev_max = np.array(debug_pgpe_history["stdev_max"])
    stdev_sensory_mean = np.array(debug_pgpe_history["stdev_sensory_mean"])
    stdev_sensory_min = np.array(debug_pgpe_history["stdev_sensory_min"])
    stdev_sensory_max = np.array(debug_pgpe_history["stdev_sensory_max"])

    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), dpi=PLOT_DPI)
    axes[0].plot(generations, center_norm, color="#3D405B", linewidth=2.0)
    axes[0].set_title("DEBUG: PGPE center norm")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("L2 norm")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(generations, stdev_mean, color="#81B29A", linewidth=2.0, label="mean")
    axes[1].plot(generations, stdev_min, color="#E07A5F", linewidth=1.5, label="min")
    axes[1].plot(generations, stdev_max, color="#3D405B", linewidth=1.5, label="max")
    axes[1].set_title("DEBUG: PGPE stdev (all dims)")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("stdev")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend()

    axes[2].plot(generations, stdev_sensory_mean, color="#81B29A", linewidth=2.0, label="mean")
    axes[2].plot(generations, stdev_sensory_min, color="#E07A5F", linewidth=1.5, label="min")
    axes[2].plot(generations, stdev_sensory_max, color="#3D405B", linewidth=1.5, label="max")
    axes[2].set_title("DEBUG: PGPE stdev (sensory-cue dims)")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("stdev")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, DEBUG_PGPE_PARAMS_FILENAME))
    plt.close(figure)


def _save_debug_pgpe_fitness_plot(plot_dir, debug_pgpe_history):
    generations = np.array(debug_pgpe_history["generation"])
    fitness_mean = np.array(debug_pgpe_history["fitness_mean"])
    fitness_max = np.array(debug_pgpe_history["fitness_max"])
    fitness_std = np.array(debug_pgpe_history["fitness_std"])

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), dpi=PLOT_DPI)
    axes[0].plot(generations, fitness_mean, color="#81B29A", linewidth=2.0, label="mean")
    axes[0].plot(generations, fitness_max, color="#3D405B", linewidth=2.0, label="max")
    axes[0].set_title("DEBUG: sampled population fitness")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Fitness")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    axes[1].plot(generations, fitness_std, color="#E07A5F", linewidth=2.0)
    axes[1].set_title("DEBUG: sampled population fitness std")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Fitness std")
    axes[1].grid(True, alpha=0.2)

    figure.tight_layout()
    figure.savefig(_prefixed_path(plot_dir, DEBUG_PGPE_FITNESS_FILENAME))
    plt.close(figure)


# ==== 3) PLOTTING HELPERS ======================================================
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


def _sort_record_by_fitness(record):
    """Return a copy of record with all per-network arrays sorted best-to-worst by fitness."""
    order = np.argsort(record["fitness"].numpy())[::-1].copy()
    sorted_record = dict(record)
    for key in ("decisions_by_run", "crashed_by_run", "rewarded_by_run",
                "big_reward_by_run", "sensory_cue_by_run", "correct_arm_by_run"):
        sorted_record[key] = record[key][order]
    return sorted_record


def _decision_category_matrix(record):
    """Encode each run's outcome into one of the 10 categories (see DECISION_LABELS):
    0=. 1=x 2=Lx 3=Rx 4=lx 5=rx 6=L 7=R 8=l 9=r
    """
    decisions = record["decisions_by_run"].numpy()
    crashed = record["crashed_by_run"].numpy()
    rewarded = record["rewarded_by_run"].numpy()
    correct_arm = record["correct_arm_by_run"].numpy()

    left = decisions == 0
    right = decisions == 1
    turned = decisions != -1

    categories = np.zeros(decisions.shape, dtype=np.int32)
    categories[crashed & ~turned] = 1                                  # x
    categories[crashed & left & correct_arm] = 2                       # Lx
    categories[crashed & right & correct_arm] = 3                      # Rx
    categories[crashed & left & ~correct_arm] = 4                      # lx
    categories[crashed & right & ~correct_arm] = 5                     # rx
    categories[rewarded & left & correct_arm] = 6                      # L
    categories[rewarded & right & correct_arm] = 7                     # R
    categories[rewarded & left & ~correct_arm] = 8                     # l
    categories[rewarded & right & ~correct_arm] = 9                    # r
    return categories


def _draw_decisions_panel(axis, matrix, title, cmap):
    """Render a single fitness-sorted decision heatmap panel."""
    im = axis.imshow(matrix, cmap=cmap, interpolation="nearest", vmin=0, vmax=N_DECISION_CATEGORIES - 1, aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("Run index")
    axis.set_ylabel("Network (best→worst)")
    return im


def _save_decisions_plot(plot_dir, tracked_records):
    """Save side-by-side heatmaps for first and last tracked generations, sorted by fitness."""
    first_matrix = _decision_category_matrix(_sort_record_by_fitness(tracked_records[0]))
    last_matrix = _decision_category_matrix(_sort_record_by_fitness(tracked_records[-1]))

    cmap = ListedColormap(DECISION_COLORS)
    figure = plt.figure(figsize=(18, 9), dpi=PLOT_DPI)
    grid = figure.add_gridspec(nrows=2, ncols=2, height_ratios=[20, 1], hspace=0.28, wspace=0.12)
    ax0 = figure.add_subplot(grid[0, 0])
    ax1 = figure.add_subplot(grid[0, 1])
    colorbar_axis = figure.add_subplot(grid[1, :])

    im0 = _draw_decisions_panel(ax0, first_matrix, f"Generation {tracked_records[0]['generation']}", cmap)
    _draw_decisions_panel(ax1, last_matrix, f"Generation {tracked_records[-1]['generation']}", cmap)

    colorbar = figure.colorbar(im0, cax=colorbar_axis, orientation="horizontal", ticks=np.arange(0, N_DECISION_CATEGORIES, 1))
    colorbar.ax.set_xticklabels(DECISION_LABELS)
    figure.suptitle("Decisions: first vs last tracked generation (sorted by fitness)")
    figure.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])
    figure.savefig(_prefixed_path(plot_dir, DECISIONS_FILENAME))
    plt.close(figure)


def _save_all_decisions_plot(plot_dir, tracked_records, tracked_generations):
    """Save a grid of decision heatmaps for every tracked generation, sorted by fitness."""
    n_plots = len(tracked_records)
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)

    cmap = ListedColormap(DECISION_COLORS)

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
        matrix = _decision_category_matrix(_sort_record_by_fitness(record))
        ax = figure.add_subplot(grid[row, col])
        im = ax.imshow(matrix, cmap=cmap, interpolation="nearest", vmin=0, vmax=N_DECISION_CATEGORIES - 1, aspect="auto")
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
    """Save a grouped bar plot: one group per tracked generation, with one bar per event
    (x, Lx, Rx, lx, rx, L, R, l, r) in each group, showing what % of that generation's
    events each event type accounted for."""
    event_labels = DECISION_LABELS[1:]  # exclude "." (not a real event, just padding)
    event_colors = DECISION_COLORS[1:]  # same event -> color mapping as the decision heatmaps
    n_events = len(event_labels)
    n_gens = len(tracked_records)

    counts = np.zeros((n_gens, n_events), dtype=int)
    for g_idx, record in enumerate(tracked_records):
        matrix = _decision_category_matrix(record)  # counts don't depend on fitness sort order
        for e_idx in range(n_events):
            counts[g_idx, e_idx] = int((matrix == e_idx + 1).sum())

    # percentage of that generation's events (the 9 real event types only -- "." padding
    # is excluded from both the numerator and the denominator), so each generation's bars
    # sum to 100% regardless of how many runs actually completed.
    totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0) * 100.0

    fig_w = max(12.0, n_events * n_gens * 0.35)
    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(fig_w, 6), dpi=PLOT_DPI)

    group_width = 0.8
    bar_width = group_width / n_events
    x_base = np.arange(n_gens)

    for e_idx in range(n_events):
        offset = (e_idx - (n_events - 1) / 2) * bar_width
        axis.bar(
            x_base + offset, percentages[:, e_idx], width=bar_width,
            color=event_colors[e_idx], edgecolor="#333333", linewidth=0.3,
            label=event_labels[e_idx],
        )

    axis.set_xticks(x_base)
    axis.set_xticklabels([f"gen {g}" for g in tracked_generations])
    axis.set_xlabel("Generation")
    axis.set_ylabel("% of events in generation")
    axis.set_title("Event distribution (%) across tracked generations")
    axis.grid(True, axis="y", alpha=0.2)
    axis.legend(ncol=min(n_events, 9), fontsize=8)
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
    """Save all-generation line plot for mean, median, and best training reward (task A, pre-L1)."""
    generations = np.array(reward_evolution["generation"])
    mean_tr = np.array(reward_evolution["training_reward_mean"])
    median_tr = np.array(reward_evolution["training_reward_median"])
    best_tr = np.array(reward_evolution["training_reward_best"])

    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=PLOT_DPI)
    axis.plot(generations, mean_tr, color=REWARD_EVOLUTION_COLORS[0], linewidth=2.0, label="mean")
    axis.plot(generations, median_tr, color=REWARD_EVOLUTION_COLORS[1], linewidth=2.0, label="median")
    axis.plot(generations, best_tr, color=REWARD_EVOLUTION_COLORS[2], linewidth=2.0, label="best")
    axis.set_title("Training reward evolution across all generations (task A, pre-L1)")
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


def _save_all_plots(run_name, history, debug_pgpe_history):
    """Create output folder and save all tracking plots."""
    plot_dir = PLOTS_ROOT / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    tracked_generations = history["tracked_generations"]
    tracked_records = history["tracked_records"]
    reward_evolution = history["reward_evolution"]
    colors = _generation_colors(tracked_generations)

    _save_decisions_plot(plot_dir, tracked_records)
    _save_all_decisions_plot(plot_dir, tracked_records, tracked_generations)
    _save_event_counts_plot(plot_dir, tracked_records, tracked_generations, colors)
    _save_reward_hist_plot(plot_dir, tracked_records, tracked_generations, colors)
    _save_frobenius_plot(plot_dir, tracked_records, tracked_generations, colors)
    _save_weight_distribution_plot(plot_dir, tracked_records, tracked_generations, colors)
    _save_reward_evolution_plot(plot_dir, reward_evolution)
    _save_training_reward_evolution_plot(plot_dir, reward_evolution)
    _save_l1_evolution_plot(plot_dir, reward_evolution)
    _save_sensory_cue_plot(plot_dir, tracked_records)
    _save_debug_pgpe_params_plot(plot_dir, debug_pgpe_history)
    _save_debug_pgpe_fitness_plot(plot_dir, debug_pgpe_history)
    print(f"\nSaved plots to: {plot_dir}")


# ==== 4) EVOLUTION RUN ==========================================================
torch.manual_seed(MASTER_SEED)
torch.cuda.manual_seed_all(MASTER_SEED)
torch.use_deterministic_algorithms(True)

noise_generator = torch.Generator(device=DEVICE)
noise_generator.manual_seed(NOISE_SEED)
reward_generator = torch.Generator(device=DEVICE)
reward_generator.manual_seed(REWARD_SEED)

center_init = torch.zeros(GENOME_LENGTH, device=DEVICE)

objective = functools.partial(
    fitness_function,
    device=DEVICE,
    noise_generator=noise_generator,
    reward_generator=reward_generator,
    l1_lambda=L1_LAMBDA,
)

configure_printing(
    total_generations=NUM_GENERATIONS,
    print_interval=TRACKED_PER_INTERVAL,
    max_networks_preview=MAX_NETWORKS_PREVIEW,
    max_runs_preview=MAX_RUNS_PREVIEW,
    hist_bin_width=HIST_BIN_WIDTH,
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

StdOutLogger(searcher)
debug_pgpe_history = _init_debug_pgpe_history()
for _ in range(NUM_GENERATIONS):
    searcher.step()
    history_snapshot = get_printing_history()
    _debug_collect_pgpe_history(searcher, history_snapshot["reward_evolution"], debug_pgpe_history)

history = get_printing_history()
_save_all_plots(RUN_NAME, history, debug_pgpe_history)

print("\nFinal searcher status:")
print(searcher.status)
print(f"\nDEBUG: sensory-cue genome dimensions tracked: {int(DEBUG_SENSORY_CUE_GENOME_INDICES.numel())}/{GENOME_LENGTH}")