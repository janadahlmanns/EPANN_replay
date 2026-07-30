"""Run PGPE with tracking output and save end-of-run plots."""

# ==== 1) RNG DETERMINISM + PATH SETUP ==========================================
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger
from matplotlib.colors import ListedColormap

from sim_core.fitness_printing import configure_printing, fitness_function_printing, get_printing_history
from sim_core.genome_codec import GENOME_LENGTH

# ==== 2) CONSTANTS / USER INPUTS ===============================================
RUN_NAME = "NAME"
DEVICE = "cuda"
MASTER_SEED = 0
NOISE_SEED = 1
REWARD_SEED = 2

NUM_GENERATIONS = 200
SEARCH_POPSIZE = 200           
RADIUS_INIT = 50.0            # radius of the initial search hypersphere in genome space (GENOME_LENGTH-dim), sweep/ optimize
MAX_SPEED = RADIUS_INIT / 15  # evotorch's rule of thumb from the ClipUp paper: max_speed = radius / 15.0, adjust the 15.0 to optimize
CENTER_LEARNING_RATE = MAX_SPEED / 2  # this is the step size in the ClipUp paper
STDEV_LEARNING_RATE = 0.1
MOMENTUM = 0.9  

L1_LAMBDA = 1e-3

TRACKED_PRINT_INTERVAL = 200
MAX_NETWORKS_PREVIEW = 6
MAX_RUNS_PREVIEW = 20
HIST_BIN_WIDTH = 1
PLOT_DPI = 180

PLOTS_ROOT = Path("C:/EPANN_replay/data/plots")
DECISIONS_FILENAME = "decisions.png"
REWARD_HIST_FILENAME = "reward_hist.png"
FROBENIUS_FILENAME = "frobenius.png"
REWARD_EVOLUTION_FILENAME = "reward_evolution.png"
SENSORY_CUE_FILENAME = "sensory_cues.png"
REWARD_EVOLUTION_COLORS = ["#E07A5F", "#3D405B", "#81B29A"]
PALETTE_COLORS = ["#E07A5F", "#3D405B", "#81B29A", "#F2CC8F", "#F4F1DE"]

DECISION_COLORS = [
    "#ffffff",  # .
    "#e3e3e3",  # x
    "#9a9a9a",  # X
    "#f3b4b4",  # l
    "#b30000",  # L
    "#bcd9ff",  # r
    "#003d99",  # R
]
DECISION_LABELS = [".", "x", "X", "l", "L", "r", "R"]


# ==== 3) PLOTTING HELPERS ======================================================
def _generation_colors(tracked_generations):
    """Create light-gray to black colors for tracked generations."""
    count = len(tracked_generations)
    gray_values = np.linspace(0.8, 0.0, count)
    colors = []
    for gray in gray_values:
        colors.append((gray, gray, gray, 1.0))
    return colors


def _decision_category_matrix(record):
    """Encode decision symbols to integer categories for heatmap plotting."""
    decisions = record["decisions_by_run"].numpy()
    crashed = record["crashed_by_run"].numpy()
    rewarded = record["rewarded_by_run"].numpy()
    big_reward = record["big_reward_by_run"].numpy()

    categories = np.zeros(decisions.shape, dtype=np.int32)
    categories[(crashed) & (decisions == -1)] = 1
    categories[(crashed) & (decisions != -1)] = 2
    categories[(~crashed) & (rewarded) & (~big_reward) & (decisions == 0)] = 3
    categories[(~crashed) & (rewarded) & (big_reward) & (decisions == 0)] = 4
    categories[(~crashed) & (rewarded) & (~big_reward) & (decisions == 1)] = 5
    categories[(~crashed) & (rewarded) & (big_reward) & (decisions == 1)] = 6
    return categories


def _save_decisions_plot(plot_dir, tracked_records):
    """Save side-by-side heatmaps for first and last tracked generations."""
    first_record = tracked_records[0]
    last_record = tracked_records[-1]
    first_matrix = _decision_category_matrix(first_record)
    last_matrix = _decision_category_matrix(last_record)

    cmap = ListedColormap(DECISION_COLORS)
    figure = plt.figure(figsize=(18, 9), dpi=PLOT_DPI)
    grid = figure.add_gridspec(nrows=2, ncols=2, height_ratios=[20, 1], hspace=0.28, wspace=0.12)
    axes = np.array([
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
    ])
    colorbar_axis = figure.add_subplot(grid[1, :])

    im0 = axes[0].imshow(first_matrix, cmap=cmap, interpolation="nearest", vmin=0, vmax=6, aspect="auto")
    axes[0].set_title(f"Generation {first_record['generation']}")
    axes[0].set_xlabel("Run index")
    axes[0].set_ylabel("Network index")

    axes[1].imshow(last_matrix, cmap=cmap, interpolation="nearest", vmin=0, vmax=6, aspect="auto")
    axes[1].set_title(f"Generation {last_record['generation']}")
    axes[1].set_xlabel("Run index")
    axes[1].set_ylabel("Network index")

    colorbar = figure.colorbar(im0, cax=colorbar_axis, orientation="horizontal", ticks=np.arange(0, 7, 1))
    colorbar.ax.set_xticklabels(DECISION_LABELS)
    figure.suptitle("Decisions: full table heatmaps (first vs last tracked generation)")
    figure.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])
    figure.savefig(plot_dir / DECISIONS_FILENAME)
    plt.close(figure)


def _save_reward_hist_plot(plot_dir, tracked_records, tracked_generations, colors):
    """Save overlapping line histograms for tracked-generation reward distributions."""
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

    axis.set_title("Reward distribution across tracked generations")
    axis.set_xlabel("Reward / fitness")
    axis.set_ylabel("Distribution density")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / REWARD_HIST_FILENAME)
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
    figure.savefig(plot_dir / FROBENIUS_FILENAME)
    plt.close(figure)


def _save_reward_evolution_plot(plot_dir, reward_evolution):
    """Save all-generation line plot for mean, median, and best population reward."""
    generations = np.array(reward_evolution["generation"])
    mean_eval = np.array(reward_evolution["mean_eval"])
    median_eval = np.array(reward_evolution["median_eval"])
    pop_best_eval = np.array(reward_evolution["pop_best_eval"])

    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=PLOT_DPI)
    axis.plot(generations, mean_eval, color=REWARD_EVOLUTION_COLORS[0], linewidth=2.0, label="mean_eval")
    axis.plot(generations, median_eval, color=REWARD_EVOLUTION_COLORS[1], linewidth=2.0, label="median_eval")
    axis.plot(generations, pop_best_eval, color=REWARD_EVOLUTION_COLORS[2], linewidth=2.0, label="pop_best_eval")
    axis.set_title("Reward evolution across all generations")
    axis.set_xlabel("Generation")
    axis.set_ylabel("Reward / fitness")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / REWARD_EVOLUTION_FILENAME)
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
    figure.savefig(plot_dir / SENSORY_CUE_FILENAME)
    plt.close(figure)


def _save_all_plots(run_name, history):
    """Create output folder and save all tracking plots."""
    plot_dir = PLOTS_ROOT / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    tracked_generations = history["tracked_generations"]
    tracked_records = history["tracked_records"]
    reward_evolution = history["reward_evolution"]
    colors = _generation_colors(tracked_generations)

    _save_decisions_plot(plot_dir, tracked_records)
    _save_reward_hist_plot(plot_dir, tracked_records, tracked_generations, colors)
    _save_frobenius_plot(plot_dir, tracked_records, tracked_generations, colors)
    _save_reward_evolution_plot(plot_dir, reward_evolution)
    _save_sensory_cue_plot(plot_dir, tracked_records)
    print(f"\nSaved plots to: {plot_dir}")


# ==== 4) EVOLUTION RUN ==========================================================
torch.manual_seed(MASTER_SEED)
torch.cuda.manual_seed_all(MASTER_SEED)
torch.use_deterministic_algorithms(True)

noise_generator = torch.Generator(device=DEVICE)
noise_generator.manual_seed(NOISE_SEED)
reward_generator = torch.Generator(device=DEVICE)
reward_generator.manual_seed(REWARD_SEED)

center_init = torch.randn(GENOME_LENGTH, device=DEVICE)

objective = functools.partial(
    fitness_function_printing,
    device=DEVICE,
    noise_generator=noise_generator,
    reward_generator=reward_generator,
    l1_lambda=L1_LAMBDA,
)

configure_printing(
    total_generations=NUM_GENERATIONS,
    print_interval=TRACKED_PRINT_INTERVAL,
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
searcher.run(NUM_GENERATIONS)

history = get_printing_history()
_save_all_plots(RUN_NAME, history)

print("\nFinal searcher status:")
print(searcher.status)
