"""Post-hoc behavioral summary: given one or more folders of run_evolution.py output
(either a single run-output folder, or -- the usual case -- a folder of GROUP folders,
each holding several run-output folders, e.g. a parameter sweep's data/<experiment>/
root), writes one CSV row per individual run summarizing its swept parameters, final
fitness, cue usage, and maze-solving strategy, plus a few summary plots -- all saved
next to each other at the given folder's own root.

Every number here is read from files run_evolution.py already writes per run (its own
config copy, event_counts.csv, input_weighing.csv, reward_evolution.csv) -- no .h5
access, so this works on any output folder regardless of whether composite_plot.py has
ever been run over it.

Swept-parameter columns aren't guessed from folder/file names (which are sweep-specific
abbreviations, see helpers/generate_configs.py) -- they're detected by diffing every
processed run's own saved config: whichever keys differ across the batch (besides the
per-run seeds) become parameter columns, so this script works unchanged on any future
sweep's folder.

"Strategy tag" is a semicolon-joined set of independently-triggered qualifiers (not a
forced single category) -- e.g. "often crashes before turning; cue-following;
right-biased" -- built from a handful of orthogonal behavioral fractions (see METRICS
section) so compound/asymmetric strategies remain describable instead of being forced
into one lossy label. Thresholds are named constants below; adjust and rerun as needed.

Usage: python -m analysis.summarize_strategies <folder> [<folder> ...]
"""

# ==== 1) IMPORTS =================================================================
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this only ever calls savefig(), never plt.show()
import matplotlib.pyplot as plt
import numpy as np

from analysis.csv_export import write_csv
from analysis.decision_plotting import DECISION_LABELS, grid_dims

# ==== 2) CONSTANTS / USER INPUTS =================================================
RUN_DIR_NAME_PATTERN = re.compile(r"^(.*)_(\d{8}-\d{6})$")  # "<run_name>_<YYYYMMDD-HHMMSS>"
SEED_KEYS = {"master_seed", "noise_seed", "reward_seed", "test_seed", "weight_init_seed"}
                                    # excluded from swept-parameter detection -- these
                                    # vary run-to-run by design (see generate_configs.py)
                                    # but aren't a "swept parameter" in the user-facing sense

SUMMARY_CSV_FILENAME = "strategy_summary.csv"
FITNESS_BY_PARAM_FILENAME = "fitness_by_parameter.png"
CUE_USAGE_BY_PARAM_FILENAME = "cue_usage_by_parameter.png"
TAG_FREQUENCY_FILENAME = "strategy_tag_frequency.png"

CUE_COLORS = {"context": "#E07A5F", "sensory": "#81B29A", "reward": "#3D405B"}  # matches
                                    # run_evolution.py's input_weighing palette (context/reward/sensory order there)

# ---- "cue used" boolean threshold: mean ablation importance (reward lost when that
# cue is clipped to zero) above this counts as "used". Context/reward importance are
# heavily bimodal around 0 in practice (most configs ignore them entirely, some rely on
# them a lot); sensory importance is almost always well above this. See conversation
# for the distribution this was picked against -- adjust freely. ----
CUE_USED_THRESHOLD = 1.0

# ---- strategy-tag thresholds, each independent (see module docstring) ----
COMPLETION_HIGH_THRESHOLD = 0.7
COMPLETION_LOW_THRESHOLD = 0.3
TURN_RATE_RARE_THRESHOLD = 0.3      # below this: "rarely turns"
TURN_RATE_REDUCED_THRESHOLD = 0.6   # between RARE and this: "often crashes before turning"
TURN_RATE_ALWAYS_THRESHOLD = 0.95   # above this: "always turns"
TURN_ACCURACY_CUE_FOLLOWING_THRESHOLD = 0.85
TURN_ACCURACY_ANTI_CUE_THRESHOLD = 0.15
TURN_ACCURACY_AGNOSTIC_BAND = (0.35, 0.65)   # chance-level arm choice
DIRECTION_BIAS_THRESHOLD = 0.85     # right_frac_of_turns above this: "right-biased"; below (1 - this): "left-biased"
POST_TURN_SURVIVAL_THRESHOLD = 0.3  # below this (given enough turns to matter): "crashes after ... turns"
MIN_TURN_RATE_FOR_SURVIVAL_TAG = 0.05   # correct_rate/incorrect_rate must clear this for the crash-after-turn tags to fire

if len(sys.argv) < 2:
    raise ValueError("Usage: python -m analysis.summarize_strategies <folder> [<folder> ...]")
TOP_FOLDERS = [Path(arg) for arg in sys.argv[1:]]


# ==== 3) RUN-DIRECTORY DISCOVERY (same convention as analysis/composite_plot.py) ==
def _natural_sort_key(name):
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", name)]


def _is_run_output_folder(folder):
    return any(RUN_DIR_NAME_PATTERN.match(path.name) for path in folder.iterdir() if path.is_dir())


def _find_run_dirs(top_folder):
    """Every run-output folder found either directly inside top_folder, or one level
    deeper inside each of top_folder's group subfolders -- whichever shape top_folder
    turns out to have. Returns a list of (group_folder_name, run_dir) pairs."""
    if _is_run_output_folder(top_folder):
        candidates = [top_folder]
    else:
        candidates = sorted(
            (path for path in top_folder.iterdir() if path.is_dir()),
            key=lambda path: _natural_sort_key(path.name),
        )
    if not candidates:
        raise ValueError(f"No subfolders found in {top_folder}")

    run_dirs = []
    for group_folder in candidates:
        if not group_folder.is_dir() or not _is_run_output_folder(group_folder):
            print(f"Skipping {group_folder.name} -- not a run-output folder (no '<stem>_<timestamp>' subfolders)")
            continue
        for run_dir in sorted(
            (p for p in group_folder.iterdir() if p.is_dir() and RUN_DIR_NAME_PATTERN.match(p.name)),
            key=lambda p: _natural_sort_key(p.name),
        ):
            run_dirs.append((group_folder.name, run_dir))
    if not run_dirs:
        raise ValueError(f"No run-output folders found under {top_folder}")
    return run_dirs


# ==== 4) PER-RUN DATA LOADING ====================================================
def _load_config(run_dir, run_name):
    """The exact config this run used -- run_evolution.py copies it into the run
    folder verbatim as "<run_name>.json"."""
    config_path = run_dir / f"{run_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _read_csv_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    with open(path, newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _last_generation_event_fractions(run_dir, run_name):
    """Fraction of trials ending in each DECISION_LABELS category, at the last
    tracked generation. Counts are summed across every phase_type/run_range group
    present at that generation before dividing -- robust to multi-phase paradigms
    (e.g. the flex paradigm) where several groups can share one generation number."""
    rows = _read_csv_rows(run_dir / f"{run_name}_event_counts.csv")
    last_generation = max(int(row["generation"]) for row in rows)
    totals = defaultdict(float)
    for row in rows:
        if int(row["generation"]) == last_generation:
            totals[row["event_label"]] += float(row["count"])
    total_count = sum(totals.values())
    if total_count == 0:
        raise ValueError(f"Zero total event count at generation {last_generation} in {run_dir}")
    return {label: totals.get(label, 0.0) / total_count for label in DECISION_LABELS}, last_generation


def _last_row(rows):
    return max(rows, key=lambda row: int(row["generation"]))


# ==== 5) BEHAVIORAL METRICS + STRATEGY TAG =======================================
def _derived_metrics(fractions):
    """Orthogonal behavioral fractions computed from the 10 raw event-category
    fractions -- see module docstring. epsilon avoids 0/0 when a rate a ratio is
    conditioned on (e.g. turn_rate) is itself zero."""
    eps = 1e-9
    reward_rate = fractions["L"] + fractions["R"] + fractions["l"] + fractions["r"]
    turn_rate = 1.0 - fractions["x"]
    left_rate = fractions["Lx"] + fractions["lx"] + fractions["L"] + fractions["l"]
    right_rate = fractions["Rx"] + fractions["rx"] + fractions["R"] + fractions["r"]
    correct_rate = fractions["Lx"] + fractions["Rx"] + fractions["L"] + fractions["R"]
    incorrect_rate = fractions["lx"] + fractions["rx"] + fractions["l"] + fractions["r"]
    return {
        "reward_rate": reward_rate,
        "crash_rate": 1.0 - reward_rate,
        "pre_turn_crash": fractions["x"],
        "post_turn_crash": (1.0 - reward_rate) - fractions["x"],
        "turn_rate": turn_rate,
        "right_frac_of_turns": right_rate / (turn_rate + eps),
        "correct_rate": correct_rate,
        "incorrect_rate": incorrect_rate,
        "turn_accuracy": correct_rate / (turn_rate + eps),
        "completion_given_turn": reward_rate / (turn_rate + eps),
        "correct_turn_survival": (fractions["L"] + fractions["R"]) / (correct_rate + eps),
        "incorrect_turn_survival": (fractions["l"] + fractions["r"]) / (incorrect_rate + eps),
        "big_reward_rate": fractions["L"] + fractions["R"],
        "small_reward_rate": fractions["l"] + fractions["r"],
    }


def _strategy_tag(m):
    """Semicolon-joined list of independently-triggered qualifiers -- see module
    docstring for why this isn't a single forced category."""
    tags = []

    if m["reward_rate"] > COMPLETION_HIGH_THRESHOLD:
        tags.append("high completion")
    elif m["reward_rate"] < COMPLETION_LOW_THRESHOLD:
        tags.append("low completion")

    if m["turn_rate"] < TURN_RATE_RARE_THRESHOLD:
        tags.append("rarely turns")
    elif m["turn_rate"] < TURN_RATE_REDUCED_THRESHOLD:
        tags.append("often crashes before turning")
    elif m["turn_rate"] > TURN_RATE_ALWAYS_THRESHOLD:
        tags.append("always turns")

    if m["turn_accuracy"] > TURN_ACCURACY_CUE_FOLLOWING_THRESHOLD:
        tags.append("cue-following")
    elif m["turn_accuracy"] < TURN_ACCURACY_ANTI_CUE_THRESHOLD:
        tags.append("anti-cue")
    elif TURN_ACCURACY_AGNOSTIC_BAND[0] <= m["turn_accuracy"] <= TURN_ACCURACY_AGNOSTIC_BAND[1]:
        tags.append("cue-agnostic (chance-level)")

    if m["right_frac_of_turns"] > DIRECTION_BIAS_THRESHOLD:
        tags.append("right-biased")
    elif m["right_frac_of_turns"] < (1.0 - DIRECTION_BIAS_THRESHOLD):
        tags.append("left-biased")

    if m["correct_rate"] > MIN_TURN_RATE_FOR_SURVIVAL_TAG and m["correct_turn_survival"] < POST_TURN_SURVIVAL_THRESHOLD:
        tags.append("crashes after correct turns")
    if m["incorrect_rate"] > MIN_TURN_RATE_FOR_SURVIVAL_TAG and m["incorrect_turn_survival"] < POST_TURN_SURVIVAL_THRESHOLD:
        tags.append("crashes after wrong turns")

    return "; ".join(tags) if tags else "no dominant pattern"


# ==== 6) ONE ROW PER RUN ==========================================================
def _detect_swept_parameters(configs):
    """Config keys (besides SEED_KEYS) whose value differs across at least two of the
    given configs, in first-seen order -- these become the CSV's parameter columns.
    Not hardcoded to any particular sweep's SWEEP_PARAMS, so this works unchanged for
    any future sweep folder (see module docstring)."""
    first_values = {}
    varying_keys = []
    for config in configs:
        for key, value in config.items():
            if key in SEED_KEYS:
                continue
            if key not in first_values:
                first_values[key] = value
            elif value != first_values[key] and key not in varying_keys:
                varying_keys.append(key)
    # preserve first-config key order rather than "varying_keys" discovery order
    ordered = [key for key in next(iter(configs)).keys() if key in varying_keys]
    return ordered


def _build_row(group_folder, run_dir, config, swept_params):
    run_name = RUN_DIR_NAME_PATTERN.match(run_dir.name).group(1)
    config_idx = run_name[len(group_folder) + 1:]  # run_name is "<group_folder>_<idx>" by construction

    fractions, last_generation = _last_generation_event_fractions(run_dir, run_name)
    metrics = _derived_metrics(fractions)
    tag = _strategy_tag(metrics)

    input_weighing_last = _last_row(_read_csv_rows(run_dir / f"{run_name}_input_weighing.csv"))
    reward_evolution_last = _last_row(_read_csv_rows(run_dir / f"{run_name}_reward_evolution.csv"))
    context_importance = float(input_weighing_last["context_importance_mean"])
    sensory_importance = float(input_weighing_last["sensory_importance_mean"])
    reward_importance = float(input_weighing_last["reward_importance_mean"])

    row = {
        "group_folder": group_folder,
        "run_name": run_name,
        "config_idx": config_idx,
    }
    for param in swept_params:
        row[param] = config[param]
    row.update({
        "last_generation": last_generation,
        "mean_eval": float(reward_evolution_last["mean_eval"]),
        "pop_best_eval": float(reward_evolution_last["pop_best_eval"]),
        "context_importance": context_importance,
        "sensory_importance": sensory_importance,
        "reward_importance": reward_importance,
        "context_used": context_importance > CUE_USED_THRESHOLD,
        "sensory_used": sensory_importance > CUE_USED_THRESHOLD,
        "reward_used": reward_importance > CUE_USED_THRESHOLD,
    })
    for label in DECISION_LABELS:
        row[f"frac_{label}"] = fractions[label]
    row.update(metrics)
    row["strategy_tag"] = tag
    return row


# ==== 7) SUMMARY PLOTS ============================================================
def _save_fitness_by_parameter(rows, swept_params, output_root):
    if not swept_params:
        return
    n_rows_grid, n_cols_grid = grid_dims(len(swept_params))
    figure, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(5.5 * n_cols_grid, 4.5 * n_rows_grid), squeeze=False)

    for idx, param in enumerate(swept_params):
        axis = axes[idx // n_cols_grid][idx % n_cols_grid]
        values = sorted({row[param] for row in rows}, key=lambda v: (isinstance(v, str), v))
        data = [[row["mean_eval"] for row in rows if row[param] == value] for value in values]
        axis.boxplot(data, tick_labels=[str(v) for v in values], showmeans=True)
        axis.set_title(param)
        axis.set_xlabel(param)
        axis.set_ylabel("mean_eval (last generation)")
        axis.grid(True, alpha=0.2)

    for spare in range(len(swept_params), n_rows_grid * n_cols_grid):
        axes[spare // n_cols_grid][spare % n_cols_grid].set_visible(False)

    figure.suptitle("Final-generation fitness by swept parameter (marginalized over every other parameter)")
    figure.tight_layout()
    figure.savefig(output_root / FITNESS_BY_PARAM_FILENAME, dpi=150)
    plt.close(figure)


def _save_cue_usage_by_parameter(rows, swept_params, output_root):
    if not swept_params:
        return
    n_rows_grid, n_cols_grid = grid_dims(len(swept_params))
    figure, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(5.5 * n_cols_grid, 4.5 * n_rows_grid), squeeze=False)
    cue_keys = ["context_used", "sensory_used", "reward_used"]

    for idx, param in enumerate(swept_params):
        axis = axes[idx // n_cols_grid][idx % n_cols_grid]
        values = sorted({row[param] for row in rows}, key=lambda v: (isinstance(v, str), v))
        x_positions = np.arange(len(values))
        bar_width = 0.8 / len(cue_keys)
        for cue_offset, cue_key in enumerate(cue_keys):
            cue_name = cue_key.split("_")[0]
            fractions = [
                np.mean([row[cue_key] for row in rows if row[param] == value])
                for value in values
            ]
            axis.bar(x_positions + cue_offset * bar_width, fractions, width=bar_width,
                     label=cue_name, color=CUE_COLORS[cue_name])
        axis.set_xticks(x_positions + bar_width, [str(v) for v in values])
        axis.set_ylim(0, 1)
        axis.set_title(param)
        axis.set_ylabel(f"fraction of runs with cue used (importance > {CUE_USED_THRESHOLD})")
        axis.grid(True, alpha=0.2, axis="y")

    axes[0][0].legend()
    for spare in range(len(swept_params), n_rows_grid * n_cols_grid):
        axes[spare // n_cols_grid][spare % n_cols_grid].set_visible(False)

    figure.suptitle("Cue usage by swept parameter (marginalized over every other parameter)")
    figure.tight_layout()
    figure.savefig(output_root / CUE_USAGE_BY_PARAM_FILENAME, dpi=150)
    plt.close(figure)


def _save_tag_frequency(rows, output_root):
    """How often each individual qualifier appears across every run's strategy_tag
    (a run contributes one count per qualifier it triggered, since tags combine
    several) -- shows which degenerate behaviors dominate this batch overall."""
    counts = defaultdict(int)
    for row in rows:
        for qualifier in row["strategy_tag"].split("; "):
            counts[qualifier] += 1

    qualifiers = sorted(counts, key=lambda q: counts[q])
    figure, axis = plt.subplots(figsize=(8, 0.4 * len(qualifiers) + 1.5))
    axis.barh(qualifiers, [counts[q] for q in qualifiers], color="#3D405B")
    axis.set_xlabel(f"number of runs (out of {len(rows)})")
    axis.set_title("Strategy-tag qualifier frequency")
    axis.grid(True, alpha=0.2, axis="x")
    figure.tight_layout()
    figure.savefig(output_root / TAG_FREQUENCY_FILENAME, dpi=150)
    plt.close(figure)


# ==== 8) MAIN EXECUTION ===========================================================
def _summarize_folder(top_folder):
    run_dirs = _find_run_dirs(top_folder)
    configs = []
    for group_folder, run_dir in run_dirs:
        run_name = RUN_DIR_NAME_PATTERN.match(run_dir.name).group(1)
        configs.append(_load_config(run_dir, run_name))
    swept_params = _detect_swept_parameters(configs)

    rows = [
        _build_row(group_folder, run_dir, config, swept_params)
        for (group_folder, run_dir), config in zip(run_dirs, configs)
    ]

    write_csv(top_folder / SUMMARY_CSV_FILENAME, rows)
    _save_fitness_by_parameter(rows, swept_params, top_folder)
    _save_cue_usage_by_parameter(rows, swept_params, top_folder)
    _save_tag_frequency(rows, top_folder)
    print(f"Summarized {len(rows)} run(s) ({len(swept_params)} swept parameter(s): {swept_params}) to: {top_folder}")


if __name__ == "__main__":
    for folder in TOP_FOLDERS:
        _summarize_folder(folder)
