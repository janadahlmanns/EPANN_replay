"""
Loads a saved genome and plots it for visual inspection:
    W, A, B, C, D   [pop, N, N]       one PNG per tensor, two heatmap panes
                                       (one per individual in INDIVIDUALS_TO_PLOT)
    beta            [pop, N]          one PNG, single heatmap (individuals x neurons)
    eta             [pop, 1]          one PNG, dot plot (one point per individual)
    M               [pop, N, N, N]    one PNG, 3D scatter of the non-zero (k, i, j)
                                       entries for individual b=0 only
"""

# ==== 1. IMPORTS ============================================================
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

# ==== 2. CONSTANTS / USER INPUTS ===========================================
INPUT_PATH = "data/test_genome.pt"
OUTPUT_DIR = "data/plots"

INDIVIDUALS_TO_PLOT = [0, 1]     # which pop indices get their own heatmap pane
M_INDIVIDUAL = 0                 # which individual the 3D M scatter is drawn for
CMAP = "RdBu_r"                  # diverging colormap, centered at 0
TENSOR_2D_NAMES = ["W", "A", "B", "C", "D"]


# ==== 3. PLOTTING FUNCTIONS =================================================
def plot_paired_heatmaps(tensor, name, individuals, cmap, output_dir):
    """One PNG per [pop, N, N] tensor: one heatmap pane per individual, shared color scale."""
    vmax = np.abs(tensor[individuals]).max()

    fig, axes = plt.subplots(1, len(individuals), figsize=(5 * len(individuals), 4.5))
    for ax, b in zip(axes, individuals):
        im = ax.imshow(tensor[b], cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name} - individual {b}")
        ax.set_xlabel("j (pre)")
        ax.set_ylabel("i (post)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="value")
    fig.savefig(os.path.join(output_dir, f"{name}_heatmaps.png"), dpi=150)
    plt.close(fig)


def plot_single_heatmap(matrix, name, cmap, output_dir):
    """One PNG for a [pop, N] tensor (e.g. beta): single heatmap, individuals x neurons."""
    vmax = np.abs(matrix).max()

    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(name)
    ax.set_xlabel("neuron")
    ax.set_ylabel("individual")
    fig.colorbar(im, ax=ax, label="value")
    fig.savefig(os.path.join(output_dir, f"{name}_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_dot(values, name, output_dir):
    """One PNG for a [pop] tensor (e.g. eta): one dot per individual."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.scatter(np.arange(len(values)), values)
    ax.set_title(name)
    ax.set_xlabel("individual")
    ax.set_ylabel("value")
    ax.set_xticks(np.arange(len(values)))
    fig.savefig(os.path.join(output_dir, f"{name}_dotplot.png"), dpi=150)
    plt.close(fig)


def plot_M_3d_scatter(M_individual, name, output_dir):
    """One PNG: 3D scatter of the non-zero (k, i, j) entries of M for one individual."""
    k, i, j = np.nonzero(M_individual)
    values = M_individual[k, i, j]
    colors = ["tab:blue" if v > 0 else "tab:red" for v in values]
    sizes = np.abs(values) / np.abs(values).max() * 200 + 40

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(k, i, j, c=colors, s=sizes)
    ax.set_title(name)
    ax.set_xlabel("k (modulating source)")
    ax.set_ylabel("i (post)")
    ax.set_zlabel("j (pre)")
    fig.savefig(os.path.join(output_dir, f"{name}_3d_scatter.png"), dpi=150)
    plt.close(fig)


# ==== 4. MAIN EXECUTION =====================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    genome = torch.load(INPUT_PATH, map_location="cpu")

    for name in TENSOR_2D_NAMES:
        plot_paired_heatmaps(genome[name].numpy(), name, INDIVIDUALS_TO_PLOT, CMAP, OUTPUT_DIR)

    plot_single_heatmap(genome["beta"].numpy(), "beta", CMAP, OUTPUT_DIR)
    plot_dot(genome["eta"].numpy().flatten(), "eta", OUTPUT_DIR)
    plot_M_3d_scatter(genome["M"][M_INDIVIDUAL].numpy(), f"M_individual{M_INDIVIDUAL}", OUTPUT_DIR)

    print(f"Saved plots to {OUTPUT_DIR}/")