"""Post-hoc composite (facet) plot builder: given a folder containing several
run_evolution.py output subfolders (each named "<config_stem>_<timestamp>"), builds
cross-config facet plots directly from whatever subfolders happen to be sitting in the
given folder: two fresh post-hoc paradigm test facets (per-paradigm-variant decisions,
and a top-10%-by-fitness comparison across variants) plus input cue importance.

The paradigm-test facets are genuinely fresh simulations, not a replay of anything
run_evolution.py itself recorded: each config's saved final-population genomes are run
once per paradigm variant, from fresh zero state/fresh initial weights, through ONLY
that variant (never a per-individual random draw among variants, unlike the real
evolutionary evaluation -- see sim_core/fitness.py's run_fixed_paradigm_test). This is
what lets a flexible-paradigm run (e.g. "paradigm": ["trainA, 50", "trainB, 50"]) be
checked for genuine paradigm-generalization -- whether the same evolved genomes do well
on BOTH tasks -- rather than each individual only ever having been evaluated on
whichever single variant it happened to draw during evolution. Never mutates the saved
genomes: online plasticity still runs within each fresh rollout (eta acts normally), but
no evolutionary update of any kind happens here. This is manual/exploratory analysis,
deliberately kept out of run_evolution.py/run_batch.py's automatic end-of-run pipeline.

If the given folder's own subfolders AREN'T run-output folders themselves (none of
them match "<config_stem>_<timestamp>"), it's treated as a folder of GROUP folders
instead: composite plots get built separately inside each of its subfolders that IS a
run-output folder, one call standing in for however many you'd otherwise run by hand.

Usage: python -m analysis.composite_plot <folder>
"""

# ==== 1) IMPORTS =================================================================
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this only ever calls savefig(), never plt.show(), and a
                        # remote GPU server reached over VPN may have no display/Tk at all
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from analysis.csv_export import write_csv
from analysis.decision_plotting import (
    DECISION_LABELS,
    N_DECISION_CATEGORIES,
    decision_category_matrix,
    draw_decisions_panel,
    grid_dims,
    sort_by_fitness,
)
from analysis.results_io import load_results_h5, results_filename
from sim_core import constants
from sim_core.fitness import run_fixed_paradigm_test
from sim_core.paradigm import parse_paradigm_variants

# ==== 2) CONSTANTS / USER INPUTS =================================================
PLOT_DPI = 180
POSTHOC_DECISIONS_FACET_PREFIX = "posthoc_decisions_facet_"  # + <variant slug> + ".png"/".csv"
FITNESS_TEST_FILENAME = "fitness_test.png"
FITNESS_TEST_CSV_FILENAME = "fitness_test.csv"
TOP_FRACTION = 0.1  # fitness_test facet: fraction of the final population (by evolved
                     # fitness, rounded UP to the next whole genome) whose post-hoc test
                     # performance gets boxplotted -- Jana's call, 2026-09-21
FITNESS_TEST_CMAP = "RdYlGn"  # each box's fill: red (low mean) -> green (high mean),
                               # scaled across every box in the WHOLE facet (all panels,
                               # all variants) -- Jana's call, 2026-09-21
INPUT_WEIGHING_FACET_FILENAME = "input_weighing_facet.png"
INPUT_WEIGHING_FACET_CSV_FILENAME = "input_weighing_facet.csv"
REWARD_EVOLUTION_COLORS = ["#E07A5F", "#3D405B", "#81B29A"]  # matches run_evolution.py's palette

POSTHOC_DEVICE = "cpu"  # manual analysis tool, not performance-critical -- avoids needing a
                         # device CLI arg / CUDA availability just to rebuild these plots
# Dedicated seeds for this post-hoc test only -- never touch a run's own training-time
# noise/reward/weight_init streams (this is an independent diagnostic probe, not a replay
# of anything evolution itself saw). Re-seeded identically before EVERY paradigm variant
# tested for a given genome (see _fresh_generators), so every variant starts from the
# identical noise/reward/initial-weight draws and differs ONLY in which task's context/
# sensory rule the paradigm applies -- isolates task-generalization from initial-weight
# luck, per Jana's call on 2026-09-21.
POSTHOC_NOISE_SEED = 90210
POSTHOC_REWARD_SEED = 90211
POSTHOC_WEIGHT_INIT_SEED = 90212

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


def _is_run_output_folder(folder):
    """True if folder directly contains run_evolution.py output subfolders (any child
    matching "<config_stem>_<YYYYMMDD-HHMMSS>") -- i.e. composite plots can be built
    for it directly, as opposed to it being a folder OF such folders."""
    return any(RUN_DIR_NAME_PATTERN.match(path.name) for path in folder.iterdir() if path.is_dir())


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


# ==== 4) CROSS-CONFIG FACET: POST-HOC PARADIGM TEST ==============================
def _load_config_and_genome(run_dir, stem):
    """One run's config (unpacked from its results.h5) plus its saved final-population
    genome tensors (M/A/B/C/D/beta/eta, as plain numpy -- constants.N-sized reshaping
    happens later, once constants is configured for THIS run) and their evolved fitness
    (numpy [pop]), used only to pick the fitness-sorted row order for the facet panel."""
    results = load_results_h5(run_dir / results_filename(stem))
    config = json.loads(results["meta"]["_attrs"]["config_json"])
    final_population = results["genome"]["final_population"]
    genome_numpy = {name: final_population[name] for name in ("M", "A", "B", "C", "D", "beta", "eta")}
    fitness = final_population["fitness"]
    return config, genome_numpy, fitness


def _configure_constants_for(config):
    """(Re-)apply this config's network layout + sim constants -- constants is global
    mutable state (see sim_core/constants.py), so this must be called again before
    simulating EVERY run_dir's genome, even within the same group: a hyperparameter
    sweep can vary any of these fields between configs (e.g. Jana's max-normalization
    softening sweep on 2026-09). Mirrors runners/run_evolution.py's own setup exactly."""
    constants.configure_network(n_neurons=config["n_neurons"])
    constants.configure(
        dt=config["dt"], tau=config["tau"], noise_std=config["noise_std"],
        straight_thresh=config["straight_thresh"], big_reward=config["big_reward"],
        small_reward=config["small_reward"], crash_penalty=config["crash_penalty"],
        turn_reward_big=config["turn_reward_big"], turn_reward_small=config["turn_reward_small"],
        tau_hebb_mult=config["tau_hebb_mult"], ma_span=config["ma_span"], weight_clamp=config["weight_clamp"],
    )


def _genome_to_tensors(genome_numpy, config):
    """numpy genome dict -> torch tensors on POSTHOC_DEVICE, with eta zeroed when this
    config evolved with plasticity off -- mirrors fitness.py's evaluate_generation."""
    genome = {name: torch.tensor(array, dtype=torch.float32, device=POSTHOC_DEVICE) for name, array in genome_numpy.items()}
    if not config["evo_plasticity_on"]:
        genome["eta"] = torch.zeros_like(genome["eta"])
    return genome


def _paradigm_variant_labels(config):
    """Config's raw "paradigm" field, normalized to a list of one label string per
    variant -- same order sim_core.paradigm.parse_paradigm_variants uses, so index i
    here always matches parse_paradigm_variants(...)[i]."""
    paradigm_field = config["paradigm"]
    return [paradigm_field] if isinstance(paradigm_field, str) else list(paradigm_field)


def _fresh_generators(device):
    """New RNG streams for one post-hoc probe, always re-seeded from the same fixed
    POSTHOC_*_SEED constants -- see their module-level comment for why (same draws
    across every paradigm variant tested for a given genome, on purpose)."""
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(POSTHOC_NOISE_SEED)
    reward_generator = torch.Generator(device=device)
    reward_generator.manual_seed(POSTHOC_REWARD_SEED)
    weight_init_generator = torch.Generator(device=device)
    weight_init_generator.manual_seed(POSTHOC_WEIGHT_INIT_SEED)
    return noise_generator, reward_generator, weight_init_generator


def _slugify(label):
    """Paradigm label -> filesystem/CSV-safe slug, e.g. "trainA, 50" -> "trainA_50"."""
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def _run_posthoc_tests(config_stems, run_dirs, output_root):
    """Load every run's config/genome/evolved-fitness and run EACH paradigm variant's
    fresh post-hoc test EXACTLY once per (config, variant) -- shared by both facet
    builders below (_save_posthoc_decisions_facets, _save_fitness_test_facet) so a
    population-sized simulation never runs twice for the same (config, variant) pair.
    Every config in the group must list the identical set of paradigm variants, in the
    same order (fails loudly otherwise -- neither facet has a sensible way to line up
    mismatched variant sets across configs).

    Returns (configs, fitnesses, variant_labels, results): results is
    {(stem, variant_idx): (tracking, reward)}, reward a numpy [pop] array of unregularized
    (no L1) training+replay reward -- see sim_core.fitness.run_fixed_paradigm_test."""
    configs, fitnesses = {}, {}
    variant_labels = None
    results = {}
    for stem, run_dir in zip(config_stems, run_dirs):
        config, genome_numpy, fitness = _load_config_and_genome(run_dir, stem)
        configs[stem], fitnesses[stem] = config, fitness

        labels = _paradigm_variant_labels(config)
        if variant_labels is None:
            variant_labels = labels
        elif labels != variant_labels:
            raise ValueError(
                f"Config {stem!r}'s paradigm variants {labels!r} don't match the group's "
                f"{variant_labels!r} -- composite_plot.py's post-hoc paradigm test needs every "
                f"config in {output_root} to list the identical set of paradigm variants, in "
                "the same order."
            )

        _configure_constants_for(config)
        genome = _genome_to_tensors(genome_numpy, config)
        for variant_idx, variant_phases in enumerate(parse_paradigm_variants(config["paradigm"])):
            noise_generator, reward_generator, weight_init_generator = _fresh_generators(POSTHOC_DEVICE)
            tracking, reward = run_fixed_paradigm_test(
                genome, variant_phases, POSTHOC_DEVICE, noise_generator, reward_generator,
                weight_init_generator, config["evo_context_cues_on"], config["evo_sensory_cues_on"],
            )
            results[(stem, variant_idx)] = (tracking, reward.cpu().numpy())

    return configs, fitnesses, variant_labels, results


def _save_posthoc_decisions_facets(config_stems, fitnesses, variant_labels, results, output_root, group_label):
    """One facet plot PER PARADIGM VARIANT listed in the group's configs: each panel is
    one config's saved final-population genomes run fresh through ONLY that variant (see
    _run_posthoc_tests), sorted by each genome's own evolved fitness -- same encoding as
    run_evolution.py's all_decisions.png."""
    n_panels = len(config_stems)
    n_rows, n_cols = grid_dims(n_panels)

    for variant_idx, variant_label in enumerate(variant_labels):
        figure = plt.figure(figsize=(8.0 * n_cols, 6.0 * n_rows + 0.7), dpi=PLOT_DPI)
        grid = figure.add_gridspec(nrows=n_rows + 1, ncols=n_cols, height_ratios=[6.0] * n_rows + [0.7], hspace=0.45, wspace=0.18)

        im_ref = None
        csv_rows = []
        for idx, stem in enumerate(config_stems):
            tracking, _ = results[(stem, variant_idx)]

            (order,) = sort_by_fitness(fitnesses[stem])
            matrix = decision_category_matrix(
                tracking["decisions_by_run"].cpu().numpy()[order],
                tracking["crashed_by_run"].cpu().numpy()[order],
                tracking["rewarded_by_run"].cpu().numpy()[order],
                tracking["correct_arm_by_run"].cpu().numpy()[order],
            )

            row, col = divmod(idx, n_cols)
            axis = figure.add_subplot(grid[row, col])
            im = draw_decisions_panel(axis, matrix, stem)
            if im_ref is None:
                im_ref = im

            n_networks, n_runs = matrix.shape
            for rank in range(n_networks):
                for run_index in range(n_runs):
                    code = int(matrix[rank, run_index])
                    csv_rows.append({
                        "config_stem": stem,
                        "paradigm_variant": variant_label,
                        "network_rank": rank,
                        "run_index": run_index,
                        "category_code": code,
                        "category_label": DECISION_LABELS[code],
                    })

        for spare in range(n_panels, n_rows * n_cols):
            row, col = divmod(spare, n_cols)
            figure.add_subplot(grid[row, col]).set_visible(False)

        colorbar_axis = figure.add_subplot(grid[n_rows, :])
        colorbar = figure.colorbar(im_ref, cax=colorbar_axis, orientation="horizontal", ticks=np.arange(0, N_DECISION_CATEGORIES, 1))
        colorbar.ax.set_xticklabels(DECISION_LABELS)
        figure.suptitle(f"Post-hoc paradigm test ({variant_label}) -- {group_label} (final population, sorted by evolved fitness)")
        # bbox_inches="tight" at save time instead of figure.tight_layout(): tight_layout()
        # doesn't support the colorbar's gridspec-placed Axes and warns every run
        slug = _slugify(variant_label)
        figure.savefig(output_root / f"{POSTHOC_DECISIONS_FACET_PREFIX}{slug}.png", bbox_inches="tight")
        plt.close(figure)

        write_csv(output_root / f"{POSTHOC_DECISIONS_FACET_PREFIX}{slug}.csv", csv_rows)


def _save_fitness_test_facet(config_stems, fitnesses, variant_labels, results, output_root, group_label):
    """One panel per config: box plots of the fitness REACHED (see _run_posthoc_tests'
    reward -- unregularized training+replay reward, no L1) by the top TOP_FRACTION of the
    final population (selected by evolved fitness, count rounded UP to the next whole
    genome), one box per paradigm variant, side by side -- shows whether evolution's own
    best individuals actually perform well on EVERY paradigm variant, not just whichever
    one they happened to be evaluated on that generation. Every panel shares the same
    y-axis range/ticks, and every box is filled by its own mean value against a single
    red (low) -> green (high) color scale spanning every box in the WHOLE facet -- both
    only meaningful/comparable because they're locked across the whole facet, not
    per-panel."""
    n_panels = len(config_stems)
    n_rows, n_cols = grid_dims(n_panels)

    # ---- first pass: gather every panel's box data once, so the shared y-axis range
    # and the shared mean-value color scale can both be fixed before any panel is drawn ----
    panel_box_data = {}
    panel_top_n = {}
    all_values = []
    all_means = []
    for stem in config_stems:
        fitness = fitnesses[stem]
        (order,) = sort_by_fitness(fitness)
        n_top = math.ceil(TOP_FRACTION * len(fitness))
        top_indices = order[:n_top]
        panel_top_n[stem] = n_top

        box_data = []
        for variant_idx in range(len(variant_labels)):
            _, reward = results[(stem, variant_idx)]
            top_reward = reward[top_indices]
            box_data.append(top_reward)
            all_values.extend(top_reward.tolist())
            all_means.append(float(top_reward.mean()))
        panel_box_data[stem] = box_data

    color_norm = Normalize(vmin=min(all_means), vmax=max(all_means))
    cmap = matplotlib.colormaps[FITNESS_TEST_CMAP]

    figure = plt.figure(figsize=(max(4.0, 1.6 * len(variant_labels)) * n_cols, 5.0 * n_rows + 0.7), dpi=PLOT_DPI)
    grid = figure.add_gridspec(nrows=n_rows + 1, ncols=n_cols, height_ratios=[5.0] * n_rows + [0.5], hspace=0.45, wspace=0.25)

    csv_rows = []
    axes = []
    for idx, stem in enumerate(config_stems):
        box_data = panel_box_data[stem]

        row, col = divmod(idx, n_cols)
        axis = figure.add_subplot(grid[row, col])
        axes.append(axis)
        boxplot = axis.boxplot(box_data, tick_labels=variant_labels, patch_artist=True)
        for patch, values in zip(boxplot["boxes"], box_data):
            patch.set_facecolor(cmap(color_norm(float(values.mean()))))
        axis.set_title(f"{stem} (top {panel_top_n[stem]} of {len(fitnesses[stem])})")
        axis.set_ylabel("Fitness reached")
        axis.set_ylim(min(all_values), max(all_values))
        axis.grid(True, axis="y", alpha=0.2)

        for variant_idx, variant_label in enumerate(variant_labels):
            for rank, value in zip(range(panel_top_n[stem]), box_data[variant_idx]):
                csv_rows.append({
                    "config_stem": stem,
                    "paradigm_variant": variant_label,
                    "network_rank": rank,
                    "fitness_reached": float(value),
                })

    # lock every panel's y-ticks to the FIRST panel's -- same ylim already makes matplotlib's
    # default locator agree, but this makes the lock explicit rather than incidental
    shared_yticks = axes[0].get_yticks()
    for axis in axes:
        axis.set_yticks(shared_yticks)
        axis.set_ylim(min(all_values), max(all_values))

    for spare in range(n_panels, n_rows * n_cols):
        row, col = divmod(spare, n_cols)
        figure.add_subplot(grid[row, col]).set_visible(False)

    colorbar_axis = figure.add_subplot(grid[n_rows, :])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=cmap), cax=colorbar_axis, orientation="horizontal",
    )
    colorbar.set_label("Box mean fitness reached")

    figure.suptitle(f"Post-hoc fitness test -- {group_label} (top {TOP_FRACTION:.0%} by evolved fitness, one box per paradigm variant)")
    figure.savefig(output_root / FITNESS_TEST_FILENAME, bbox_inches="tight")
    plt.close(figure)

    write_csv(output_root / FITNESS_TEST_CSV_FILENAME, csv_rows)


# ==== 5) CROSS-CONFIG FACET: INPUT CUE IMPORTANCE ================================
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
    csv_rows = []
    for idx, (stem, run_dir) in enumerate(zip(config_stems, run_dirs)):
        cue_importance = load_results_h5(run_dir / results_filename(stem))["history"]["cue_importance"]
        row, col = divmod(idx, n_cols)
        _draw_input_weighing_panel(axes[row, col], cue_importance, stem)

        for i, generation in enumerate(cue_importance["generation"]):
            csv_rows.append({
                "config_stem": stem,
                "generation": int(generation),
                "context_importance_mean": cue_importance["context_importance_mean"][i],
                "context_importance_min": cue_importance["context_importance_min"][i],
                "context_importance_max": cue_importance["context_importance_max"][i],
                "sensory_importance_mean": cue_importance["sensory_importance_mean"][i],
                "sensory_importance_min": cue_importance["sensory_importance_min"][i],
                "sensory_importance_max": cue_importance["sensory_importance_max"][i],
                "reward_importance_mean": cue_importance["reward_importance_mean"][i],
                "reward_importance_min": cue_importance["reward_importance_min"][i],
                "reward_importance_max": cue_importance["reward_importance_max"][i],
            })

    for spare in range(n_panels, n_rows * n_cols):
        row, col = divmod(spare, n_cols)
        axes[row, col].set_visible(False)

    axes[0, 0].legend()
    figure.suptitle(f"Input cue importance -- {group_label} (reward lost when cue is ablated)")
    figure.tight_layout()
    figure.savefig(output_root / INPUT_WEIGHING_FACET_FILENAME)
    plt.close(figure)

    write_csv(output_root / INPUT_WEIGHING_FACET_CSV_FILENAME, csv_rows)


# ==== 6) MAIN EXECUTION ===========================================================
def _build_composite_plots(folder, group_label):
    """Build every facet plot for one run-output folder."""
    config_stems, run_dirs = _find_run_dirs(folder)
    _, fitnesses, variant_labels, results = _run_posthoc_tests(config_stems, run_dirs, folder)
    _save_posthoc_decisions_facets(config_stems, fitnesses, variant_labels, results, folder, group_label)
    _save_fitness_test_facet(config_stems, fitnesses, variant_labels, results, folder, group_label)
    _save_input_weighing_facet(config_stems, run_dirs, folder, group_label)
    print(f"Saved composite facet plots for {len(config_stems)} run(s) to: {folder}")


if _is_run_output_folder(TOP_FOLDER):
    _build_composite_plots(TOP_FOLDER, GROUP_LABEL)
else:
    # TOP_FOLDER holds several GROUPS' folders (e.g. one sweep's worth), not run
    # folders directly -- build composite plots inside each group folder in turn,
    # skipping anything that isn't a run-output folder either rather than erroring
    # out partway through everyone else's plots.
    subfolders = sorted(
        (path for path in TOP_FOLDER.iterdir() if path.is_dir()),
        key=lambda path: _natural_sort_key(path.name),
    )
    if not subfolders:
        raise ValueError(f"No subfolders found in {TOP_FOLDER}")

    for subfolder in subfolders:
        if not _is_run_output_folder(subfolder):
            print(f"Skipping {subfolder.name} -- not a run-output folder (no '<stem>_<timestamp>' subfolders)")
            continue
        _build_composite_plots(subfolder, subfolder.name)
