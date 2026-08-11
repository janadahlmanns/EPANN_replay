"""Runs run_evolution.py once per config file in a batch folder, each as its own
subprocess (clean isolation -- no shared CUDA/RNG/module state between runs), then
builds cross-config facet plots (final-generation decisions, input cue importance)
from the h5 results each run wrote.

Configs are grouped by filename stem-minus-trailing-integer (e.g. ep_n40_0.json,
ep_n40_1.json, ... all belong to group "ep_n40"). A group needs 2+ members to
count -- a config whose stem is unique doesn't form a group on its own.
  - 2+ groups found: each group's runs + its own facet-plot pair land in their own
    subfolder, data/<experiment_name>/<group>/; any leftover non-grouped configs
    (unique stems) land directly in data/<experiment_name>/ with their own shared
    facet-plot pair.
  - 0 or 1 groups found: no subfolders at all -- every run + one shared facet-plot
    pair land directly in data/<experiment_name>/, same as if nothing were grouped.

Usage: python run_batch.py <experiment_name>

RUN_IN_PARALLEL controls how many config chains run at once (each chain still runs
its own configs one after another). Only raise this above 1 when you know the GPU
has headroom for it -- e.g. running 2 side by side, like manually running
run_batch.py in two terminals on two halves of the folder.
"""

# ==== 1) IMPORTS =================================================================
import os
import re
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

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
BATCH_FOLDER = PROJECT_ROOT / "configs" / "batch_to_run78"  # fixed location -- put all input json to be run into this folder, script then runs all consecutively
RUN_EVOLUTION_SCRIPT = PROJECT_ROOT / "runners" / "run_evolution.py"
DATA_ROOT = PROJECT_ROOT / "data"
RUN_IN_PARALLEL = 5  # how many config chains to run concurrently -- only raise this if you're at
                     # the computer and sure the sims haven't grown enough to fight over GPU memory

ROOT_GROUP_KEY = ""  # sentinel group key for "no subfolder, straight into the experiment root"
GROUP_STEM_PATTERN = re.compile(r"^(.*)_(\d+)$")  # "<group>_<trailing integer>"

PLOT_DPI = 180
DECISIONS_FACET_FILENAME = "final_generation_decisions_facet.png"
INPUT_WEIGHING_FACET_FILENAME = "input_weighing_facet.png"
REWARD_EVOLUTION_COLORS = ["#E07A5F", "#3D405B", "#81B29A"]  # matches run_evolution.py's palette

if len(sys.argv) != 2:
    raise ValueError("Usage: python run_batch78.py <experiment_name>")
EXPERIMENT_NAME = sys.argv[1]
OUTPUT_ROOT = DATA_ROOT / EXPERIMENT_NAME


# ==== 3) CONFIG GROUPING =========================================================
def _natural_sort_key(path):
    """Sort key that orders embedded numbers by value, not by character (so
    ..._2 comes before ..._10, not after)."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", path.stem)]


def _group_stem(path):
    """'<group>_<trailing integer>' -> '<group>'; None if the stem has no trailing integer."""
    match = GROUP_STEM_PATTERN.match(path.stem)
    return match.group(1) if match else None


def _assign_groups(config_paths):
    """Map each config path to its group key: its _group_stem() if that stem has
    2+ members (a real group), else ROOT_GROUP_KEY. If fewer than 2 real groups
    exist overall, every path maps to ROOT_GROUP_KEY instead -- no point in
    subfoldering a single group (or none) away from everything else."""
    stems = [_group_stem(path) for path in config_paths]
    stem_counts = Counter(stem for stem in stems if stem is not None)
    real_groups = {stem for stem, count in stem_counts.items() if count >= 2}

    if len(real_groups) < 2:
        return {path: ROOT_GROUP_KEY for path in config_paths}
    return {
        path: (stem if stem in real_groups else ROOT_GROUP_KEY)
        for path, stem in zip(config_paths, stems)
    }


def _group_output_root(group_key):
    """data/<experiment_name>/ for the root group, data/<experiment_name>/<group_key>/ otherwise."""
    return OUTPUT_ROOT if group_key == ROOT_GROUP_KEY else OUTPUT_ROOT / group_key


def _group_label(group_key):
    """Human-readable label for a facet plot's title."""
    return "all configs" if group_key == ROOT_GROUP_KEY else group_key


# ==== 4) BATCH EXECUTION =========================================================
def _run_chain(chain_idx, chain_config_paths, group_of):
    for config_path in chain_config_paths:
        print(f"\n{'=' * 90}\nCHAIN {chain_idx}: {config_path.name}\n{'=' * 90}\n")
        # run_evolution.py resolves its config_name argument as configs/<name>.json,
        # so "batch_to_run/<stem>" reaches this config the same way
        config_name = f"batch_to_run78/{config_path.stem}"
        group_key = group_of[config_path]
        experiment_name = EXPERIMENT_NAME if group_key == ROOT_GROUP_KEY else f"{EXPERIMENT_NAME}/{group_key}"
        subprocess.run(
            [sys.executable, str(RUN_EVOLUTION_SCRIPT), config_name, experiment_name],
            check=True,
        )


def _resolve_run_dir(config_stem, search_root):
    """Find the (single) timestamped result folder run_evolution.py just created for
    this config. Sorts lexicographically and takes the last match so a leftover
    folder from a previous batch under the same experiment name can't get picked
    over this run's fresh one (timestamp format is lexicographically sortable)."""
    matches = sorted(search_root.glob(f"{config_stem}_*"))
    if not matches:
        raise FileNotFoundError(f"No result folder found for config '{config_stem}' under {search_root}")
    return matches[-1]


# ==== 5) CROSS-CONFIG FACET: FINAL-GENERATION DECISIONS =========================
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


# ==== 6) CROSS-CONFIG FACET: INPUT CUE IMPORTANCE ===============================
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


# ==== 7) MAIN EXECUTION ===========================================================
config_paths = sorted(BATCH_FOLDER.glob("*.json"), key=_natural_sort_key)
if not config_paths:
    raise ValueError(f"No .json config files found in {BATCH_FOLDER}")

group_of = _assign_groups(config_paths)

# Warm up imports of unsigned/native-extension libs (matplotlib, torch, ...) in a
# throwaway subprocess *before* spawning the parallel chains. Without this, two chains
# can both import them for the first time at the same instant, and Windows'
# application-control/reputation check on the never-before-seen DLLs can flake and
# block one of the concurrent loads (seen with matplotlib's _image DLL).
subprocess.run([sys.executable, "-c", "import matplotlib.pyplot, torch"], check=True)

# split into RUN_IN_PARALLEL contiguous chunks, one chain per chunk
chains = [config_paths[chain_idx::RUN_IN_PARALLEL] for chain_idx in range(RUN_IN_PARALLEL)]

threads = [
    threading.Thread(target=_run_chain, args=(chain_idx, chain_config_paths, group_of))
    for chain_idx, chain_config_paths in enumerate(chains)
    if chain_config_paths
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

print(f"\nBatch complete: {len(config_paths)} runs from {BATCH_FOLDER} across {len(threads)} parallel chain(s)")

for group_key in sorted(set(group_of.values()), key=lambda k: (k == ROOT_GROUP_KEY, k)):
    group_config_paths = [path for path in config_paths if group_of[path] == group_key]
    group_output_root = _group_output_root(group_key)
    group_label = _group_label(group_key)

    config_stems = [config_path.stem for config_path in group_config_paths]
    run_dirs = [_resolve_run_dir(stem, group_output_root) for stem in config_stems]
    _save_decisions_facet(config_stems, run_dirs, group_output_root, group_label)
    _save_input_weighing_facet(config_stems, run_dirs, group_output_root, group_label)
    print(f"Saved facet plots for {group_label} to: {group_output_root}")
