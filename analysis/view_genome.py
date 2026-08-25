"""Genome viewer: loads a run's results.h5 and renders the evolved plasticity genome
(A, B, C, D, eta, beta) for the BEST individual of the final generation, plus a
separate faceted plot for the NxNxN modulatory tensor M (one NxN heatmap per
source/gating neuron) -- so we can actually look at what evolution baked into the
plasticity rule instead of only judging it by task performance.

BEST only, deliberately -- genome/best is general across every es_method (see
analysis/results_io.py), so this viewer works unchanged no matter which search
algorithm produced the run. An earlier version also plotted the CENTER individual,
but "center" is a PGPE/Gaussian-search-distribution-only concept (genome/pgpe/center)
with no equivalent for a genuinely population-based method like Cosyne -- dropped
entirely rather than gated, so this tool stays comparable across es_methods.

A, B, C, D, eta are (N, N) with axis 0 = post-synaptic neuron i, axis 1 = pre-synaptic
neuron j (matches ctrnn.py's einsum("bij,bj->bi", W, state) convention). M is
(N, N, N) = M[k, i, j], the modulatory weight from source/gating neuron k onto synapse
(i, j); its facet heatmaps use the same (i, j) axis orientation as A/B/C/D so the two
figures read the same way.

Usage: python -m analysis.view_genome <path_to_h5_file>
"""

# ==== 1) IMPORTS =================================================================
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")  # headless -- this only ever calls savefig(), never plt.show(), and a
                        # remote GPU server reached over VPN may have no display/Tk at all
import matplotlib.pyplot as plt
import numpy as np

from analysis.decision_plotting import grid_dims
from analysis.results_io import results_filename

# ==== 2) CONSTANTS / USER INPUTS =================================================
PLOT_DPI = 180
HEATMAP_CMAP = "coolwarm"  # diverging, centered at 0 -- every genome param here is signed
HIST_BINS = 60
GENOME_LABELS = ("A", "B", "C", "D")  # order the four plasticity-coefficient panels are drawn in

if len(sys.argv) != 2:
    raise ValueError("Usage: python -m analysis.view_genome <path_to_run_dir>")
RUN_DIR = Path(sys.argv[1])  # e.g. data/temp/perfect_solver -- must contain "<RUN_DIR.name>_results.h5",
                              # same run-name/filename convention run_evolution.py's RUN_DIR always writes
STEM = RUN_DIR.name
H5_PATH = RUN_DIR / results_filename(STEM)
OUTPUT_DIR = RUN_DIR


# ==== 3) GENOME LOADING ===========================================================
def _load_genome_group(h5_file, group_name):
    """Load one genome group (genome/best) into a dict of numpy arrays, squeezing the
    leading pop=1 axis every tensor in that group is stored with."""
    group = h5_file[group_name]
    return {name: group[name][()][0] for name in ("A", "B", "C", "D", "eta", "beta", "M")}


def _load_best(h5_path):
    with h5py.File(h5_path, "r") as h5_file:
        best = _load_genome_group(h5_file, "genome/best")
        best_fitness = float(h5_file["genome/best/fitness"][()][0])
    return best, best_fitness


# ==== 4) MAIN GENOME PLOT (A, B, C, D, eta, beta) ================================
def _draw_heatmap(axis, matrix, title):
    vmax = np.abs(matrix).max()
    vmax = vmax if vmax > 0 else 1.0  # guard an all-zero panel from a degenerate 0..0 color range
    im = axis.imshow(matrix, cmap=HEATMAP_CMAP, vmin=-vmax, vmax=vmax, aspect="equal")
    axis.set_title(title)
    axis.set_xlabel("pre neuron (j)")
    axis.set_ylabel("post neuron (i)")
    return im


def _draw_histogram(axis, values, title):
    axis.hist(values.reshape(-1), bins=HIST_BINS, color="#3D405B")
    axis.set_title(title)
    axis.set_xlabel("value")
    axis.set_ylabel("count")


def _plot_main_genome_figure(genome, label, subtitle, output_path):
    """label: 'BEST' or 'CENTER'. subtitle: extra title info (e.g. fitness), or ''."""
    # height_ratios sized to each row's actual content (heatmap rows want ~square cells at this
    # width, histogram/scatter rows don't) -- a plain square figsize left a large blank gap above
    # the first row, since equal gridspec rows gave the square heatmaps far more height than width.
    figure = plt.figure(figsize=(20, 17), dpi=PLOT_DPI)
    grid = figure.add_gridspec(nrows=4, ncols=4, height_ratios=[5, 4, 5, 3], hspace=0.5, wspace=0.4)

    for col, name in enumerate(GENOME_LABELS):
        heat_axis = figure.add_subplot(grid[0, col])
        im = _draw_heatmap(heat_axis, genome[name], name)
        figure.colorbar(im, ax=heat_axis, fraction=0.046, pad=0.04)

        hist_axis = figure.add_subplot(grid[1, col])
        _draw_histogram(hist_axis, genome[name], f"{name} distribution")

    eta_heat_axis = figure.add_subplot(grid[2, 0])
    im = _draw_heatmap(eta_heat_axis, genome["eta"], "eta")
    figure.colorbar(im, ax=eta_heat_axis, fraction=0.046, pad=0.04)

    eta_hist_axis = figure.add_subplot(grid[2, 1])
    _draw_histogram(eta_hist_axis, genome["eta"], "eta distribution")

    for col in (2, 3):  # eta only needs 2 of this row's 4 slots -- hide the rest
        figure.add_subplot(grid[2, col]).set_visible(False)

    beta_axis = figure.add_subplot(grid[3, :])
    neuron_ids = np.arange(genome["beta"].shape[0])
    beta_axis.scatter(neuron_ids, genome["beta"], color="#E07A5F")
    beta_axis.set_xticks(neuron_ids)
    beta_axis.set_title("beta (tonic activation)")
    beta_axis.set_xlabel("neuron ID")
    beta_axis.set_ylabel("value")
    beta_axis.grid(True, alpha=0.2)

    title = f"{label} GENOME -- {STEM}"
    if subtitle:
        title += f" ({subtitle})"
    figure.suptitle(title, fontsize=16)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


# ==== 5) M (MODULATORY TENSOR) FACET PLOT ========================================
def _plot_M_facets(M, label, subtitle, output_path):
    """One NxN heatmap per source/gating neuron k, same (i, j) = (post, pre) axis
    orientation as the A/B/C/D heatmaps above, so the two figures read the same way."""
    n_source_neurons = M.shape[0]
    n_rows, n_cols = grid_dims(n_source_neurons)
    vmax = np.abs(M).max()
    vmax = vmax if vmax > 0 else 1.0

    figure, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(4.0 * n_cols, 4.0 * n_rows), dpi=PLOT_DPI, squeeze=False
    )
    im_ref = None
    for k in range(n_source_neurons):
        row, col = divmod(k, n_cols)
        axis = axes[row, col]
        im = axis.imshow(M[k], cmap=HEATMAP_CMAP, vmin=-vmax, vmax=vmax, aspect="equal")
        axis.set_title(f"source neuron {k}")
        axis.set_xlabel("pre neuron (j)")
        axis.set_ylabel("post neuron (i)")
        if im_ref is None:
            im_ref = im

    for spare in range(n_source_neurons, n_rows * n_cols):
        row, col = divmod(spare, n_cols)
        axes[row, col].set_visible(False)

    figure.colorbar(im_ref, ax=axes, fraction=0.02, pad=0.02)
    title = f"{label} GENOME -- M (modulatory weights) -- {STEM}"
    if subtitle:
        title += f" ({subtitle})"
    figure.suptitle(title, fontsize=16)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


# ==== 6) MAIN EXECUTION ===========================================================
BEST_GENOME, BEST_FITNESS = _load_best(H5_PATH)

_plot_main_genome_figure(BEST_GENOME, "BEST", f"fitness={BEST_FITNESS:.4f}", OUTPUT_DIR / f"{STEM}_best_genome.png")
_plot_M_facets(BEST_GENOME["M"], "BEST", f"fitness={BEST_FITNESS:.4f}", OUTPUT_DIR / f"{STEM}_best_genome_M.png")

print(f"Saved genome plots for {STEM} to: {OUTPUT_DIR}")
