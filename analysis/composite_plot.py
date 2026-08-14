"""Post-hoc composite (facet) plot builder: given a folder containing several
run_evolution.py output subfolders (each named "<config_stem>_<timestamp>"), builds
the same two cross-config facet plots run_batch.py normally builds automatically once
a group finishes -- final-generation decisions and input cue importance -- directly
from whatever subfolders happen to be sitting in the given folder.

For rebuilding those plots by hand when they're missing (a batch that crashed before
finishing, folders moved/merged after the fact, etc.) without re-running the batch.

Usage: python -m analysis.composite_plot <folder>
"""

# ==== 1) IMPORTS =================================================================
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this only ever calls savefig(), never plt.show(), and a
                        # remote GPU server reached over VPN may have no display/Tk at all
import matplotlib.pyplot as plt
import numpy as np

from analysis.decision_plotting import (
    DECISION_LABELS,
    N_DECISION_CATEGORIES,
    decision_category_matrix,
    draw_decisions_panel,
    grid_dims,
    sort_by_fitness,
)
from analysis.results_io import load_results_h5, results_filename

# ==== 2) CONSTANTS / USER INPUTS =================================================
PLOT_DPI = 180
DECISIONS_FACET_FILENAME = "final_generation_decisions_facet.png"
INPUT_WEIGHING_FACET_FILENAME = "input_weighing_facet.png"
REWARD_EVOLUTION_COLORS = ["#E07A5F", "#3D405B", "#81B29A"]  # matches run_evolution.py's palette

RUN_DIR_NAME_PATTERN = re.compile(r"^(.*)_(\d{8}-\d{6})$")  # "<config_stem>_<YYYYMMDD-HHMMSS>"

if len(sys.argv) != 2:
    raise ValueError("Usage: python -m analysis.composite_plot <folder>")
TOP_FOLDER = Path(sys.argv[1])
GROUP_LABEL = TOP_FOLDER.name


# ==== 3) RUN-DIRECTORY DISCOVERY =================================================
def _natural_sort_key(name):
    """Sort key that orders embedded numbers by value, not by character (so
    ..._2 comes before ..._10, not after) -- matches run_batch.py's config ordering."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", name)]


def _find_run_dirs(top_folder):
    """Every direct subfolder of top_folder is treated as one run's output directory,
    named "<config_stem>_<timestamp>" by run_evolution.py (the timestamp can't be
    predicted, so this has to list what's actually on disk rather than deriving it).
    Returns (config_stems, run_dirs), both sorted together by natural order of the stem."""
    subfolders = [path for path in top_folder.iterdir() if path.is_dir()]
    if not subfolders:
        raise ValueError(f"No subfolders found in {top_folder}")

    stems_and_dirs = []
    for subfolder in subfolders:
        match = RUN_DIR_NAME_PATTERN.match(subfolder.name)
        if not match:
            raise ValueError(
                f"Subfolder {subfolder.name!r} doesn't match the expected "
                f"'<config_stem>_<YYYYMMDD-HHMMSS>' naming -- is {top_folder} really "
                f"a folder of run_evolution.py output directories?"
            )
        stems_and_dirs.append((match.group(1), subfolder))

    stems_and_dirs.sort(key=lambda pair: _natural_sort_key(pair[0]))
    config_stems, run_dirs = zip(*stems_and_dirs)
    return list(config_stems), list(run_dirs)


# ==== 4) CROSS-CONFIG FACET: FINAL-GENERATION DECISIONS ==========================
def _save_decisions_facet(config_stems, run_dirs, output_root, group_label):
    """One panel per config: that config's last tracked generation's decision
    heatmap, sorted by fitness -- same encoding as run_evolution.py's all_decisions.png."""
    n_panels = len(config_stems)
    n_rows, n_cols = grid_dims(n_panels)

    figure = plt.figure(figsize=(8.0 * n_cols, 6.0 * n_rows + 0.7), dpi=PLOT_DPI)
    grid = figure.add_gridspec(nrows=n_rows + 1, ncols=n_cols, height_ratios=[6.0] * n_rows + [0.7], hspace=0.45, wspace=0.18)

    im_ref = None
    for idx, (stem, run_dir) in enumerate(zip(config_stems, run_dirs)):
        tracked = load_results_h5(run_dir / results_filename(stem))["history"]["tracked"]
        generation = int(tracked["generation"][-1])
        fitness = tracked["fitness"][-1]
        (order,) = sort_by_fitness(fitness)
        matrix = decision_category_matrix(
            tracked["decisions_by_run"][-1][order],
            tracked["crashed_by_run"][-1][order],
            tracked["rewarded_by_run"][-1][order],
            tracked["correct_arm_by_run"][-1][order],
        )

        row, col = divmod(idx, n_cols)
        axis = figure.add_subplot(grid[row, col])
        im = draw_decisions_panel(axis, matrix, f"{stem} (gen {generation})")
        if im_ref is None:
            im_ref = im

    for spare in range(n_panels, n_rows * n_cols):
        row, col = divmod(spare, n_cols)
        figure.add_subplot(grid[row, col]).set_visible(False)

    colorbar_axis = figure.add_subplot(grid[n_rows, :])
    colorbar = figure.colorbar(im_ref, cax=colorbar_axis, orientation="horizontal", ticks=np.arange(0, N_DECISION_CATEGORIES, 1))
    colorbar.ax.set_xticklabels(DECISION_LABELS)
    figure.suptitle(f"Final-generation decisions -- {group_label} (sorted by fitness)")
    # bbox_inches="tight" at save time instead of figure.tight_layout(): tight_layout()
    # doesn't support the colorbar's gridspec-placed Axes and warns every run
    figure.savefig(output_root / DECISIONS_FACET_FILENAME, bbox_inches="tight")
    plt.close(figure)


# ==== 5) CROSS-CONFIG FACET: INPUT CUE IMPORTANCE ================================
def _draw_input_weighing_panel(axis, cue_importance, title):
    """Same line-plot content as run_evolution.py's input_weighing.png, one config's worth."""
    generations = cue_importance["generation"]
    axis.plot(generations, cue_importance["context_importance_mean"], color=REWARD_EVOLUTION_COLORS[0], linewidth=2.0, label="context cue")
    axis.fill_between(generations, cue_importance["context_importance_min"], cue_importance["context_importance_max"], color=REWARD_EVOLUTION_COLORS[0], alpha=0.15)
    axis.plot(generations, cue_importance["sensory_importance_mean"], color=REWARD_EVOLUTION_COLORS[2], linewidth=2.0, label="sensory cue")
    axis.fill_between(generations, cue_importance["sensory_importance_min"], cue_importance["sensory_importance_max"], color=REWARD_EVOLUTION_COLORS[2], alpha=0.15)
    axis.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    axis.set_title(title)
    axis.set_xlabel("Generation")
    axis.set_ylabel("Reward lost when cue is clipped to zero")
    axis.grid(True, alpha=0.2)


def _save_input_weighing_facet(config_stems, run_dirs, output_root, group_label):
    """One panel per config: that config's ablation-based cue importance over tracked generations."""
    n_panels = len(config_stems)
    n_rows, n_cols = grid_dims(n_panels)

    figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8.0 * n_cols, 5.0 * n_rows), dpi=PLOT_DPI, squeeze=False)
    for idx, (stem, run_dir) in enumerate(zip(config_stems, run_dirs)):
        cue_importance = load_results_h5(run_dir / results_filename(stem))["history"]["cue_importance"]
        row, col = divmod(idx, n_cols)
        _draw_input_weighing_panel(axes[row, col], cue_importance, stem)

    for spare in range(n_panels, n_rows * n_cols):
        row, col = divmod(spare, n_cols)
        axes[row, col].set_visible(False)

    axes[0, 0].legend()
    figure.suptitle(f"Input cue importance -- {group_label} (reward lost when cue is ablated)")
    figure.tight_layout()
    figure.savefig(output_root / INPUT_WEIGHING_FACET_FILENAME)
    plt.close(figure)


# ==== 6) MAIN EXECUTION ===========================================================
config_stems, run_dirs = _find_run_dirs(TOP_FOLDER)
_save_decisions_facet(config_stems, run_dirs, TOP_FOLDER, GROUP_LABEL)
_save_input_weighing_facet(config_stems, run_dirs, TOP_FOLDER, GROUP_LABEL)
print(f"Saved composite facet plots for {len(config_stems)} run(s) to: {TOP_FOLDER}")
