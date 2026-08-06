"""Shared decision-outcome plotting: color/label scheme, category encoding, and
panel-grid helpers used by both run_evolution.py (per-run plots) and run_batch.py
(cross-config facet plots), so both stay visually consistent."""

import math

import numpy as np
from matplotlib.colors import ListedColormap

# ==== 1) DECISION-OUTCOME COLOR SCHEME ==========================================
# Encodes each maze run's outcome into one of 10 categories (see DECISION_LABELS).
# Color design: lightness encodes crash (light) vs. maze-end reward (dark); hue
# encodes turn direction (red=left, blue=right; gray/neutral = no turn); saturation
# encodes whether the chosen arm was correct (high) or not (low).
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
DECISION_CMAP = ListedColormap(DECISION_COLORS)


# ==== 2) CATEGORY ENCODING + SORTING ============================================
def decision_category_matrix(decisions, crashed, rewarded, correct_arm):
    """Encode each run's outcome into one of the 10 categories (see DECISION_LABELS):
    0=. 1=x 2=Lx 3=Rx 4=lx 5=rx 6=L 7=R 8=l 9=r. All inputs are numpy arrays of the
    same shape (e.g. [pop, num_runs])."""
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


def sort_by_fitness(fitness, *arrays):
    """Return (order, *sorted_arrays), best-to-worst by fitness, along axis 0."""
    order = np.argsort(fitness)[::-1].copy()
    return (order, *(array[order] for array in arrays))


# ==== 3) PANEL / GRID DRAWING HELPERS ===========================================
def draw_decisions_panel(axis, matrix, title):
    """Render a single fitness-sorted decision heatmap panel onto axis."""
    im = axis.imshow(
        matrix, cmap=DECISION_CMAP, interpolation="nearest",
        vmin=0, vmax=N_DECISION_CATEGORIES - 1, aspect="auto",
    )
    axis.set_title(title)
    axis.set_xlabel("Run index")
    axis.set_ylabel("Network (best->worst)")
    return im


def grid_dims(n_panels):
    """Roughly-square (n_rows, n_cols) layout for n_panels subplots."""
    n_cols = math.ceil(math.sqrt(n_panels))
    n_rows = math.ceil(n_panels / n_cols)
    return n_rows, n_cols
