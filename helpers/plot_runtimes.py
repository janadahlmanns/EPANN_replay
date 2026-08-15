"""Grouped boxplot + jittered scatter of seconds_per_parallel_generation, from one
machine's runtime log (see runners/run_batch.py's _log_runtime) -- a quick visual for
"how many parallel chains, at what network size, is actually fastest on this machine."

Same data shown two ways, stacked: top panel groups by n_chains (n_neurons as the
sub-groups within each cluster); bottom panel groups by n_neurons (n_chains as the
sub-groups) -- two views onto the same stratified data instead of a 3D plot.

Two encodings, each consistent across BOTH panels regardless of which variable is the
primary grouping there: color = n_chains (from analysis/plot_colors.txt, overflowing
into a standard colormap if there are more n_chains values than listed colors), marker
shape = n_neurons (a fixed shape sequence, filled for the first pass through the
sequence, unfilled -- same shapes, hollow -- once that runs out).

Usage: python -m helpers.plot_runtimes <log_name>
"""

# ==== 1) IMPORTS =================================================================
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this only ever calls savefig(), never plt.show()
import matplotlib.pyplot as plt
import numpy as np

# ==== 2) CONSTANTS / USER INPUTS =================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
PLOT_COLORS_PATH = PROJECT_ROOT / "analysis" / "plot_colors.txt"

PLOT_DPI = 180
N_PALETTE_COLORS_TO_USE = 4  # analysis/plot_colors.txt's 5th color (a very pale cream) is
                              # skipped -- too faint to read against a white plot background
OVERFLOW_COLORMAP_NAME = "Dark2"  # ColorBrewer qualitative palette, used once
                                    # analysis/plot_colors.txt's colors run out
MARKER_SHAPES = ["o", "^", "s", "*", "D", "p", "h", "X"]  # cycled twice: filled, then the
                                                            # same shapes again but unfilled
JITTER_WIDTH = 0.10  # horizontal spread of scattered points around each box's x position
JITTER_SEED = 0  # fixed, purely so the same log produces a visually stable plot on reruns
BOX_WIDTH = 0.7
BOX_LINEWIDTH = 2.2
MARKER_SIZE = 45

if len(sys.argv) != 2:
    raise ValueError("Usage: python -m helpers.plot_runtimes <log_name>")
LOG_NAME = sys.argv[1]
CSV_PATH = DATA_ROOT / f"runtime_log_{LOG_NAME}.csv"
OUTPUT_PATH = DATA_ROOT / f"runtime_plot_{LOG_NAME}.png"


# ==== 3) DATA LOADING =============================================================
def _load_grouped(csv_path):
    """Returns {(n_chains, n_neurons): [seconds_per_parallel_generation, ...]}."""
    grouped = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            key = (int(row["n_chains"]), int(row["n_neurons"]))
            grouped.setdefault(key, []).append(float(row["seconds_per_parallel_generation"]))
    return grouped


# ==== 4) COLOR (n_chains) / MARKER SHAPE (n_neurons) ASSIGNMENT =================
def _load_palette_colors(path):
    with open(path, "r", encoding="utf-8") as colors_file:
        return [f"#{line.strip()}" for line in colors_file if line.strip()]


PALETTE_COLORS = _load_palette_colors(PLOT_COLORS_PATH)[:N_PALETTE_COLORS_TO_USE]
OVERFLOW_COLORMAP = plt.colormaps[OVERFLOW_COLORMAP_NAME]


def _build_color_map(values):
    """value -> hex color, from PALETTE_COLORS first, then the overflow colormap."""
    color_map = {}
    for idx, value in enumerate(values):
        if idx < len(PALETTE_COLORS):
            color_map[value] = PALETTE_COLORS[idx]
        else:
            color_map[value] = OVERFLOW_COLORMAP((idx - len(PALETTE_COLORS)) % OVERFLOW_COLORMAP.N)
    return color_map


def _build_marker_map(values):
    """value -> (shape, filled). First len(MARKER_SHAPES) values get filled markers,
    cycling through MARKER_SHAPES; the next len(MARKER_SHAPES) values reuse the same
    shapes unfilled; repeats from there if there are ever more values than that."""
    marker_map = {}
    n_shapes = len(MARKER_SHAPES)
    for idx, value in enumerate(values):
        marker_map[value] = (MARKER_SHAPES[idx % n_shapes], (idx // n_shapes) % 2 == 0)
    return marker_map


# ==== 5) GROUPED BOXPLOT PANEL ===================================================
def _grouped_boxplot(axis, grouped, primary_values, secondary_values, primary_label, chains_of, neurons_of, color_map, marker_map, rng):
    """One cluster of boxes per primary value; within each cluster, one box per
    secondary value. grouped is keyed (n_chains, n_neurons) -> list of y-values;
    chains_of/neurons_of map (primary, secondary) -> the actual n_chains/n_neurons
    value for that box, since which of primary/secondary is which varies by panel."""
    n_secondary = len(secondary_values)
    cluster_width = n_secondary + 1  # the +1 leaves a gap between clusters

    cluster_centers = []
    for cluster_idx, primary in enumerate(primary_values):
        cluster_start = cluster_idx * cluster_width
        cluster_centers.append(cluster_start + (n_secondary + 1) / 2)
        for sub_idx, secondary in enumerate(secondary_values):
            n_chains = chains_of(primary, secondary)
            n_neurons = neurons_of(primary, secondary)
            values = grouped.get((n_chains, n_neurons))
            if not values:
                continue
            position = cluster_start + sub_idx + 1
            color = color_map[n_chains]
            shape, filled = marker_map[n_neurons]

            axis.boxplot(
                [values], positions=[position], widths=BOX_WIDTH, showfliers=False,
                medianprops={"color": color, "linewidth": BOX_LINEWIDTH},
                boxprops={"color": color, "linewidth": BOX_LINEWIDTH},
                whiskerprops={"color": color, "linewidth": BOX_LINEWIDTH},
                capprops={"color": color, "linewidth": BOX_LINEWIDTH},
            )
            jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(values))
            axis.scatter(
                np.full(len(values), position) + jitter, values,
                marker=shape, s=MARKER_SIZE, zorder=3, linewidths=1.2,
                facecolors=color if filled else "none", edgecolors=color,
            )

    axis.set_xticks(cluster_centers)
    axis.set_xticklabels([str(value) for value in primary_values])
    axis.set_xlabel(primary_label)
    axis.set_ylabel("seconds per parallel generation")
    axis.grid(True, axis="y", alpha=0.2)
    axis.margins(y=0.25)  # headroom so the top-corner legends don't overlap tall boxes


def _add_color_legend(axis, color_map):
    """color -> n_chains. Only needed on a panel where n_chains ISN'T already the
    x-axis category (that panel's x tick labels already say it)."""
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=str(n_chains))
        for n_chains, color in color_map.items()
    ]
    axis.legend(handles=handles, title="n_chains", loc="upper right", fontsize=8)


def _add_shape_legend(axis, marker_map):
    """shape -> n_neurons. Only needed on a panel where n_neurons ISN'T already the
    x-axis category."""
    handles = [
        plt.Line2D(
            [0], [0], marker=shape, linestyle="", markersize=8,
            markerfacecolor="#3D405B" if filled else "none", markeredgecolor="#3D405B",
            label=str(n_neurons),
        )
        for n_neurons, (shape, filled) in marker_map.items()
    ]
    axis.legend(handles=handles, title="n_neurons", loc="upper right", fontsize=8)


# ==== 6) MAIN EXECUTION ===========================================================
grouped = _load_grouped(CSV_PATH)
n_chains_values = sorted({key[0] for key in grouped})
n_neurons_values = sorted({key[1] for key in grouped})
color_map = _build_color_map(n_chains_values)
marker_map = _build_marker_map(n_neurons_values)

rng = np.random.default_rng(JITTER_SEED)
figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(2.2 * len(n_chains_values) + 3.0, 12.0), dpi=PLOT_DPI)

_grouped_boxplot(
    axes[0], grouped, n_chains_values, n_neurons_values, "n_chains",
    chains_of=lambda primary, secondary: primary, neurons_of=lambda primary, secondary: secondary,
    color_map=color_map, marker_map=marker_map, rng=rng,
)
_add_shape_legend(axes[0], marker_map)  # n_chains is already on this panel's x-axis
axes[0].set_title(f"Runtime by parallelism, split by network size -- {LOG_NAME}")

_grouped_boxplot(
    axes[1], grouped, n_neurons_values, n_chains_values, "n_neurons",
    chains_of=lambda primary, secondary: secondary, neurons_of=lambda primary, secondary: primary,
    color_map=color_map, marker_map=marker_map, rng=rng,
)
_add_color_legend(axes[1], color_map)  # n_neurons is already on this panel's x-axis
axes[1].set_title(f"Runtime by network size, split by parallelism -- {LOG_NAME}")

figure.tight_layout()
figure.savefig(OUTPUT_PATH)
plt.close(figure)
print(f"Saved runtime plot to: {OUTPUT_PATH}")
