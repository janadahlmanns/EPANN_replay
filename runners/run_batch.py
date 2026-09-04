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

Usage: python -m runners.run_batch <config_folder> <experiment_name> <device> <n_chains> <log_name> <early_termination_enabled>

config_folder is looked up as configs/<config_folder>/ (just like batch_to_run used
to be hardcoded) -- this is what lets different machines each point at their own
folder of configs to run without touching each other's.

n_chains controls how many config chains run at once (each chain still runs its own
configs one after another). Only raise this above 1 when you know the GPU has
headroom for it -- e.g. running 2 side by side on a machine with two GPUs.

device is passed straight through to each run_evolution.py subprocess call
(overriding any "device" field in the config json -- see run_evolution.py).

early_termination_enabled is exactly "True" or "False" (fails loudly on anything else),
passed straight through to every run_evolution.py subprocess call -- see that script's
docstring/sim_core/fitness.py's _check_event_count_termination for what it controls.
"False" forces every config in this batch to run its full configured generation count
regardless of the early-termination criterion, e.g. to see whether a run that would
normally look collapsed actually recovers given more generations.

Each config is moved into <config_folder>/done/ the moment its run_evolution.py
subprocess finishes successfully -- so config_folder always reflects what's still
left to do, and rerunning run_batch.py over the same folder (e.g. after a crash)
only reprocesses whatever wasn't moved to done/ yet.

Configs can safely be moved into or out of config_folder while this script is running
(e.g. shifting load to/from another machine mid-sweep): each "pass" snapshots whatever's
in config_folder at that moment, and if a snapshotted config is gone by the time its
chain reaches it (moved elsewhere), that chain just skips it and moves on -- it does not
crash. Once a pass finishes, if config_folder still has .json files left (leftovers a
chain skipped, or new ones dropped in mid-run), another pass runs automatically, up to
MAX_PASSES times, until the folder's empty.

log_name picks which runtime log this invocation appends to (while the batch is running):
data_temp/runtime_log_<log_name>.csv (see _log_runtime), published to
DESTINATION_ROOT/runtime_log_<log_name>.csv only once the whole batch finishes (see
_publish_to_destination below). One separate file per machine/GPU on purpose -- multiple
machines appending to the SAME synced file is a real corruption risk that a same-process lock
can't protect against. Every finished config appends one row: config name, chain, n_chains,
runtime in seconds.

Each individual run_evolution.py subprocess call writes its own output to a plain LOCAL folder
(data_temp/, see that script's DATA_ROOT) for as long as it's actively creating files, then
moves that one finished, already-complete run_dir into DESTINATION_ROOT (data/, the synced
folder, e.g. FAUbox) itself as its very last step -- see run_evolution.py's docstring. This
script never sees data_temp/<experiment_name>/ at all: by the time subprocess.run() returns for
a config, DESTINATION_ROOT/<experiment_name>/ already has that config's finished run_dir in it,
which is where this script looks for it (_find_run_dir) and where it writes each group's facet
plots. Only the runtime log is buffered locally by THIS script and merged into DESTINATION_ROOT
at the very end (_publish_to_destination) -- everything else publishes itself per-run instead.
This two-step (local folder, then one clean move of the whole finished thing) exists because
writing straight into a synced folder while dozens of files are still being created for one run
raced with the sync client's own filesystem filter driver often enough to intermittently corrupt
output (empty or half-written run folders). The tradeoff: you can no longer watch a run's
progress live via the sync folder while it's still writing, only once it finishes.
"""

# ==== 1) IMPORTS =================================================================
import csv
import re
import shutil
import subprocess
import sys
import time
import threading
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")  # headless, thread-safe rendering -- facet plots are now built from inside
                        # background chain threads (see _maybe_plot_group), and the default
                        # interactive backend (e.g. TkAgg) isn't thread-safe, throwing spurious
                        # "main thread is not in main loop" errors during interpreter shutdown
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
RUN_EVOLUTION_SCRIPT = PROJECT_ROOT / "runners" / "run_evolution.py"
DATA_ROOT = PROJECT_ROOT / "data_temp"  # LOCAL folder this script buffers its own runtime log
                                         # in while the batch runs -- see _publish_to_destination()
DESTINATION_ROOT = PROJECT_ROOT / "data"  # synced folder (e.g. FAUbox) -- where every run publishes
                                           # its own finished output straight to (see run_evolution.py),
                                           # and where the runtime log is published at the very end

ROOT_GROUP_KEY = ""  # sentinel group key for "no subfolder, straight into the experiment root"
GROUP_STEM_PATTERN = re.compile(r"^(.*)_(\d+)$")  # "<group>_<trailing integer>"

PLOT_DPI = 180
DECISIONS_FACET_FILENAME = "final_generation_decisions_facet.png"
INPUT_WEIGHING_FACET_FILENAME = "input_weighing_facet.png"
REWARD_EVOLUTION_COLORS = ["#E07A5F", "#3D405B", "#81B29A"]  # matches run_evolution.py's palette

DONE_SUBFOLDER = "done"  # finished configs get moved to <BATCH_FOLDER>/done/ -- Path.glob("*.json")
                          # only matches direct children, so this is all that's needed to keep a
                          # rerun of run_batch.py over the same folder from reprocessing them

RUNTIME_LOG_HEADER = [
    "config", "n_neurons", "popsize", "chain", "n_chains", "runtime_seconds", "generations_run",
    "seconds_per_generation", "seconds_per_parallel_generation",
]

def _parse_bool_arg(value, arg_name):
    """Strict True/False CLI-argument parser -- argv values are always strings, and this
    project's style forbids silently guessing (e.g. treating any non-"False" string as
    True), so anything other than exactly "True" or "False" fails loudly."""
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{arg_name} must be exactly 'True' or 'False', got {value!r}")


if len(sys.argv) != 7:
    raise ValueError(
        "Usage: python -m runners.run_batch <config_folder> <experiment_name> <device> <n_chains> "
        "<log_name> <early_termination_enabled: True/False>"
    )
CONFIG_FOLDER_NAME = sys.argv[1]  # looked up as configs/<CONFIG_FOLDER_NAME>/, never elsewhere
EXPERIMENT_NAME = sys.argv[2]
DEVICE = sys.argv[3]  # passed straight through to every run_evolution.py subprocess call
RUN_IN_PARALLEL = int(sys.argv[4])  # how many config chains to run concurrently -- only raise this if
                                     # you're sure the target machine's GPU has headroom for it
LOG_NAME = sys.argv[5]  # picks data_temp/runtime_log_<LOG_NAME>.csv -- one file per machine, see docstring
EARLY_TERMINATION_ENABLED = _parse_bool_arg(sys.argv[6], "early_termination_enabled")  # passed straight
                                     # through to every run_evolution.py subprocess call, see docstring

BATCH_FOLDER = PROJECT_ROOT / "configs" / CONFIG_FOLDER_NAME
RUNTIME_LOG_PATH = DATA_ROOT / f"runtime_log_{LOG_NAME}.csv"
# Each run_evolution.py subprocess call publishes its OWN finished run_dir straight to
# DESTINATION_ROOT itself (see run_evolution.py) -- by the time subprocess.run() returns
# below, that run's folder already lives here, not under DATA_ROOT. So facet-plot output
# and the _find_run_dir lookups this script does both target DESTINATION_ROOT too.
OUTPUT_ROOT = DESTINATION_ROOT / EXPERIMENT_NAME


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
# Group facet plots are attempted after EVERY config finishes (not just once at the
# very end) -- see _maybe_plot_group. _PLOTTED_GROUPS/_PLOT_LOCK are this script's
# only cross-thread coordination: multiple chains can each finish the last config of
# the same group at nearly the same moment, and the lock + already-plotted check keep
# that from racing into a duplicate (or file-corrupting concurrent-write) plot.
_PLOTTED_GROUPS = set()
_PLOT_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()  # guards RUNTIME_LOG_PATH -- multiple chains can finish and append
                               # to the same shared csv at nearly the same moment


def _run_dir_pattern(config_stem):
    """Matches exactly "<config_stem>_<timestamp>" where timestamp is run_evolution.py's
    RUN_TIMESTAMP format (YYYYMMDD-HHMMSS). Anchored (fullmatch) and exact-length so a
    config stem that's a prefix of ANOTHER config's stem -- e.g. "param_test" vs
    "param_test_cosyne" -- can't have its glob swallow that other config's run folder;
    a loose f"{config_stem}_*" glob would match "param_test_cosyne_20260825-140225"
    too, since it also starts with "param_test_"."""
    return re.compile(rf"^{re.escape(config_stem)}_\d{{8}}-\d{{6}}$")


def _find_run_dir(config_stem, search_root):
    """Find the (single) timestamped result folder run_evolution.py creates for this
    config, or None if it doesn't exist (yet, or ever -- e.g. that config's run
    crashed). Sorts lexicographically and takes the last match so a leftover folder
    from a previous batch under the same experiment name can't get picked over this
    run's fresh one (timestamp format is lexicographically sortable)."""
    pattern = _run_dir_pattern(config_stem)
    matches = sorted(path for path in search_root.glob(f"{config_stem}_*") if pattern.fullmatch(path.name))
    return matches[-1] if matches else None


def _read_run_metadata(run_dir, config_stem):
    """Reads generations_run, n_neurons, and search_popsize from this run's own h5 file, in one open.
    generations_run: how many generations history/reward_evolution/generation actually
    recorded -- may be less than the config's num_generations if an early-termination
    criterion fired (see sim_core/fitness.py). n_neurons: read from the saved config's
    attrs (meta/config), not the original config file, so it's tied to exactly what
    this run used. search_popsize: read from the saved config's attrs (meta/config), not the original config file, so it's tied to exactly what this run used. All are cheap/small reads, not the whole h5 file (which can be
    large once tracked-generation snapshots are included)."""
    h5_path = run_dir / results_filename(config_stem)
    with h5py.File(h5_path, "r") as h5_file:
        generations_run = h5_file["history/reward_evolution/generation"].shape[0]
        n_neurons = int(h5_file["meta/config"].attrs["n_neurons"])
        search_popsize = int(h5_file["meta/config"].attrs["search_popsize"])
    return generations_run, n_neurons, search_popsize


def _log_runtime(config_path, n_neurons, search_popsize, chain_idx, runtime_seconds, generations_run):
    """Append one row to RUNTIME_LOG_PATH for this config's completed run. Writes the
    header only the first time this particular log file is created. The lock only
    protects against this one process's own chains racing each other -- a different
    machine writing the SAME log file at the same time is exactly what LOG_NAME (one
    file per machine) is meant to avoid; the lock can't protect a synced file across
    two separate processes on two separate machines.

    seconds_per_generation is wall-clock time per generation for THIS chain while
    contending with RUN_IN_PARALLEL-1 others for the GPU -- not comparable across
    different n_chains settings on its own, since more contention slows each chain
    down even as more total work happens per wall-clock second. Dividing by n_chains
    (seconds_per_parallel_generation) converts it into wall-clock seconds per
    generation of TOTAL work across the whole batch, which IS comparable: multiplying
    it by the sum of generations_run across every config in a batch gives that batch's
    total wall-clock runtime, regardless of how many chains it used."""
    seconds_per_generation = runtime_seconds / generations_run
    seconds_per_parallel_generation = seconds_per_generation / RUN_IN_PARALLEL
    with _LOG_LOCK:
        write_header = not RUNTIME_LOG_PATH.exists()
        RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RUNTIME_LOG_PATH, "a", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            if write_header:
                writer.writerow(RUNTIME_LOG_HEADER)
            writer.writerow([
                config_path.stem, n_neurons, search_popsize, chain_idx, RUN_IN_PARALLEL,
                round(runtime_seconds), generations_run, round(seconds_per_generation, 2),
                round(seconds_per_parallel_generation, 3),
            ])


def _maybe_plot_group(group_key, group_of, config_paths):
    """Called after every config finishes. Checks whether ALL configs in this config's
    group now have a result folder on disk, and if so renders that group's facet plots
    -- exactly once. If the group isn't complete yet (a sibling config in another chain
    hasn't finished, or never will because it crashed), this just returns quietly
    instead of raising -- so one crashed/still-running config can no longer take the
    whole batch runner down or block facet plots for every OTHER (finished) group.
    A later rerun of run_batch.py, after manually re-running whatever config was
    missing, will pick up the group's plots at that point."""
    with _PLOT_LOCK:
        if group_key in _PLOTTED_GROUPS:
            return
        group_config_paths = [path for path in config_paths if group_of[path] == group_key]
        group_output_root = _group_output_root(group_key)
        config_stems = [path.stem for path in group_config_paths]

        run_dirs = []
        for stem in config_stems:
            run_dir = _find_run_dir(stem, group_output_root)
            if run_dir is None:
                return  # not complete yet -- next config to finish will try again
            run_dirs.append(run_dir)

        group_label = _group_label(group_key)
        _save_decisions_facet(config_stems, run_dirs, group_output_root, group_label)
        _save_input_weighing_facet(config_stems, run_dirs, group_output_root, group_label)
        print(f"Saved facet plots for {group_label} to: {group_output_root}")
        _PLOTTED_GROUPS.add(group_key)


def _mark_config_done(config_path):
    """Move a just-finished config into <BATCH_FOLDER>/done/, so it's excluded from
    any future BATCH_FOLDER.glob("*.json") -- see DONE_SUBFOLDER."""
    done_dir = BATCH_FOLDER / DONE_SUBFOLDER
    done_dir.mkdir(exist_ok=True)
    shutil.move(str(config_path), str(done_dir / config_path.name))


def _format_duration(seconds):
    """Human-readable Hh Mm Ss duration, dropping leading zero units (e.g. "45s",
    "12m 03s", "2h 05m 09s")."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _run_chain(chain_idx, chain_config_paths, config_paths, group_of):
    for config_path in chain_config_paths:
        if not config_path.exists():
            # moved/removed by hand after this pass's snapshot was taken (e.g. reassigned
            # to another machine) -- skip it and keep going, rather than letting
            # run_evolution.py discover it missing and crash this whole chain
            print(f"CHAIN {chain_idx}: skipping {config_path.name} -- no longer present (moved elsewhere?)")
            continue
        print(f"\n{'=' * 90}\nCHAIN {chain_idx}: {config_path.name}\n{'=' * 90}\n")
        # run_evolution.py resolves its config_name argument as configs/<name>.json,
        # so "<CONFIG_FOLDER_NAME>/<stem>" reaches this config the same way
        config_name = f"{CONFIG_FOLDER_NAME}/{config_path.stem}"
        group_key = group_of[config_path]
        experiment_name = EXPERIMENT_NAME if group_key == ROOT_GROUP_KEY else f"{EXPERIMENT_NAME}/{group_key}"
        tick = time.monotonic()
        try:
            subprocess.run(
                [
                    sys.executable, str(RUN_EVOLUTION_SCRIPT), config_name, experiment_name, DEVICE, str(chain_idx),
                    str(EARLY_TERMINATION_ENABLED),
                ],
                check=True,
            )
            runtime_seconds = time.monotonic() - tick
            print(f"CHAIN {chain_idx}: processed {config_path.name} in {_format_duration(runtime_seconds)}")
            run_dir = _find_run_dir(config_path.stem, _group_output_root(group_key))
            generations_run, n_neurons, search_popsize = _read_run_metadata(run_dir, config_path.stem)
            _log_runtime(config_path, n_neurons, search_popsize, chain_idx, runtime_seconds, generations_run)
            _mark_config_done(config_path)
            _maybe_plot_group(group_key, group_of, config_paths)
        except Exception as error:
            # One config's subprocess crashing (or its results being unreadable right after,
            # e.g. a still-forming run_dir) must not take the rest of this chain's queue down
            # with it -- log it and move on, same spirit as the "moved elsewhere" skip above.
            # config_path is deliberately left in BATCH_FOLDER (not moved to done/), so the
            # next pass picks it up and retries it automatically.
            print(f"CHAIN {chain_idx}: {config_path.name} failed -- {error!r} -- leaving it in {BATCH_FOLDER} for the next pass")


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
    axis.plot(generations, cue_importance["reward_importance_mean"], color=REWARD_EVOLUTION_COLORS[1], linewidth=2.0, label="reward signal")
    axis.fill_between(generations, cue_importance["reward_importance_min"], cue_importance["reward_importance_max"], color=REWARD_EVOLUTION_COLORS[1], alpha=0.15)
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


# ==== 7) PUBLISH TO DESTINATION ===================================================
def _publish_to_destination():
    """Appends this session's local runtime-log rows onto
    DESTINATION_ROOT/runtime_log_<LOG_NAME>.csv (writing the header too if that file
    doesn't exist yet there), then deletes the local copy. This is the only thing left
    to publish here -- every run's own output folder already published itself straight
    to DESTINATION_ROOT as it finished (see run_evolution.py). Called once, after the
    whole batch (every pass) has finished, whether or not every config in it succeeded.
    No-ops if this session never logged anything (e.g. every config failed before
    finishing)."""
    if not RUNTIME_LOG_PATH.exists():
        return
    DESTINATION_ROOT.mkdir(parents=True, exist_ok=True)
    destination_log_path = DESTINATION_ROOT / RUNTIME_LOG_PATH.name
    write_header = not destination_log_path.exists()
    with open(RUNTIME_LOG_PATH, "r", encoding="utf-8") as local_log:
        rows = local_log.readlines()
    with open(destination_log_path, "a", newline="", encoding="utf-8") as destination_log:
        destination_log.writelines(rows if write_header else rows[1:])
    RUNTIME_LOG_PATH.unlink()
    print(f"\nPublished this session's runtime log to: {destination_log_path}")


# ==== 8) MAIN EXECUTION ===========================================================
MAX_PASSES = 5  # bounds the "reprocess whatever's left" loop below -- a config that fails
                 # the same way every pass (a real bug, not a moved/missing file) would
                 # otherwise keep the folder from ever emptying out and loop forever


def _run_one_pass(pass_num):
    """Snapshot whatever configs are currently sitting in BATCH_FOLDER, group them, and
    run them across RUN_IN_PARALLEL chains. Returns how many configs were snapshotted
    this pass (0 means there was nothing to do -- the caller should stop looping).
    Configs added to BATCH_FOLDER after this snapshot, or moved away and reappearing
    later, are simply picked up by the NEXT pass, not this one."""
    config_paths = sorted(BATCH_FOLDER.glob("*.json"), key=_natural_sort_key)
    if not config_paths:
        return 0

    group_of = _assign_groups(config_paths)
    chains = [config_paths[chain_idx::RUN_IN_PARALLEL] for chain_idx in range(RUN_IN_PARALLEL)]
    threads = [
        threading.Thread(target=_run_chain, args=(chain_idx, chain_config_paths, config_paths, group_of))
        for chain_idx, chain_config_paths in enumerate(chains)
        if chain_config_paths
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print(f"\nPass {pass_num} complete: {len(config_paths)} config(s) attempted from {BATCH_FOLDER} across {len(threads)} parallel chain(s)")

    # Every group's plots are already attempted as its configs finish (see _maybe_plot_group)
    # -- this is just a diagnostic so an unattended/multi-server run leaves a clear record
    # of which groups (and which specific configs within them) didn't complete THIS pass,
    # instead of you having to go hunt for missing folders by hand.
    all_group_keys = sorted(set(group_of.values()), key=lambda k: (k == ROOT_GROUP_KEY, k))
    incomplete_groups = [group_key for group_key in all_group_keys if group_key not in _PLOTTED_GROUPS]
    if incomplete_groups:
        print(f"{len(incomplete_groups)} group(s) not complete after pass {pass_num}:")
        for group_key in incomplete_groups:
            group_output_root = _group_output_root(group_key)
            missing_stems = [
                path.stem for path in config_paths
                if group_of[path] == group_key and _find_run_dir(path.stem, group_output_root) is None
            ]
            print(f"  {_group_label(group_key)}: missing {missing_stems}")

    return len(config_paths)


if not any(BATCH_FOLDER.glob("*.json")):
    raise ValueError(f"No .json config files found in {BATCH_FOLDER}")

# Warm up imports of unsigned/native-extension libs (matplotlib, torch, ...) in a
# throwaway subprocess *before* spawning the parallel chains. Without this, two chains
# can both import them for the first time at the same instant, and Windows'
# application-control/reputation check on the never-before-seen DLLs can flake and
# block one of the concurrent loads (seen with matplotlib's _image DLL).
subprocess.run(
    [sys.executable, "-c", "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot, torch"],
    check=True,
)

for pass_num in range(1, MAX_PASSES + 1):
    n_attempted = _run_one_pass(pass_num)
    if n_attempted == 0:
        print(f"\nNo .json config files left in {BATCH_FOLDER} -- done.")
        break
else:
    print(
        f"\nStopped after {MAX_PASSES} passes with configs still left in {BATCH_FOLDER} -- likely "
        "a config that fails the same way every pass (a real bug, not a moved file). Check it by hand."
    )

_publish_to_destination()
