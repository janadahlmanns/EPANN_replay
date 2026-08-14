"""Plot saved printing-vs-non-printing compute profiles."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this only ever calls savefig(), never plt.show(), and a
                        # remote GPU server reached over VPN may have no display/Tk at all
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "printing_compute"
BASELINE_PROFILE = DATA_DIR / "run_evolution_profile.json"
PRINTING_PROFILE = DATA_DIR / "run_evolution_printing_profile.json"
OUTPUT_OVERVIEW_PLOT = DATA_DIR / "compare_printing_compute.png"
OUTPUT_PHASES_PLOT = DATA_DIR / "compare_printing_compute_phases.png"
OUTPUT_TRAINING_PLOT = DATA_DIR / "compare_printing_compute_training_detail.png"


def _load_profile(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing profile file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _extract_totals(profile, tracked_only=None, generation_filter=None):
    rows = profile["iterations"]
    if tracked_only is not None:
        rows = [row for row in rows if bool(row["is_tracked_detail"]) == tracked_only]
    if generation_filter is not None:
        allowed = set(generation_filter)
        rows = [row for row in rows if row["generation"] in allowed]
    generations = [row["generation"] for row in rows]
    totals = [row["total_seconds"] for row in rows]
    return generations, totals


def _iteration_map(profile, tracked_only=None):
    rows = profile["iterations"]
    if tracked_only is not None:
        rows = [row for row in rows if bool(row["is_tracked_detail"]) == tracked_only]
    return {row["generation"]: row for row in rows}


def _mean_section_seconds(rows, section_names):
    values = {}
    for section_name in section_names:
        section_values = [row["sections"].get(section_name) for row in rows if section_name in row["sections"]]
        values[section_name] = _mean(section_values)
    return values


def _phase_series(rows, section_name):
    generations = [row["generation"] for row in rows if section_name in row["sections"]]
    values = [row["sections"][section_name] for row in rows if section_name in row["sections"]]
    return generations, values


def _safe_ratio(numerator, denominator):
    if denominator in (None, 0):
        return None
    if numerator is None:
        return None
    return numerator / denominator


def _compare_metadata(baseline, printing):
    differing_keys = []
    baseline_metadata = baseline.get("metadata", {})
    printing_metadata = printing.get("metadata", {})
    interesting_keys = [
        "device",
        "master_seed",
        "noise_seed",
        "reward_seed",
        "search_popsize",
        "run_name",
    ]
    for key in interesting_keys:
        if baseline_metadata.get(key) != printing_metadata.get(key):
            differing_keys.append((key, baseline_metadata.get(key), printing_metadata.get(key)))

    top_level_keys = ["total_generations", "tracked_interval"]
    for key in top_level_keys:
        if baseline.get(key) != printing.get(key):
            differing_keys.append((key, baseline.get(key), printing.get(key)))
    return differing_keys


def _annotate_bars(axis, labels, values):
    for label, value in zip(labels, values):
        if value is None:
            continue
        axis.text(label, value, f"{value:.4f}s", ha="center", va="bottom", fontsize=9)


def main():
    baseline = _load_profile(BASELINE_PROFILE)
    printing = _load_profile(PRINTING_PROFILE)

    metadata_differences = _compare_metadata(baseline, printing)
    baseline_generations, baseline_totals = _extract_totals(baseline)
    printing_generations, printing_totals = _extract_totals(printing)
    printing_detail_generations, printing_detail_totals = _extract_totals(printing, tracked_only=True)
    printing_other_generations, printing_other_totals = _extract_totals(printing, tracked_only=False)

    baseline_by_generation = _iteration_map(baseline)
    printing_by_generation = _iteration_map(printing, tracked_only=True)
    common_detail_generations = [
        generation for generation in printing_detail_generations
        if generation in baseline_by_generation
    ]
    matched_baseline_detail_totals = [
        baseline_by_generation[generation]["total_seconds"]
        for generation in common_detail_generations
    ]
    matched_printing_detail_totals = [
        printing_by_generation[generation]["total_seconds"]
        for generation in common_detail_generations
    ]
    matched_slowdown = [
        printing_total / baseline_total
        for printing_total, baseline_total in zip(matched_printing_detail_totals, matched_baseline_detail_totals)
        if baseline_total > 0.0
    ]

    baseline_detail_rows = [baseline_by_generation[generation] for generation in common_detail_generations]
    printing_detail_rows = [printing_by_generation[generation] for generation in common_detail_generations]

    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 14), dpi=180)

    axes[0].plot(baseline_generations, baseline_totals, color="#3D405B", linewidth=1.6, label="non-printing")
    axes[0].plot(printing_generations, printing_totals, color="#E07A5F", linewidth=1.6, label="printing")
    axes[0].scatter(
        printing_detail_generations,
        printing_detail_totals,
        color="#B30000",
        s=40,
        label="printing tracked-detail",
        zorder=3,
    )
    axes[0].set_title("Per-generation compute time")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Seconds")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    summary_labels = [
        "printing tracked-detail",
        "printing other",
        "non-printing all",
        "non-printing matched detail",
    ]
    summary_values = [
        _mean(printing_detail_totals),
        _mean(printing_other_totals),
        _mean(baseline_totals),
        _mean(matched_baseline_detail_totals),
    ]
    bar_values = [0.0 if value is None else value for value in summary_values]
    colors = ["#B30000", "#E07A5F", "#3D405B", "#81B29A"]
    axes[1].bar(summary_labels, bar_values, color=colors)
    _annotate_bars(axes[1], summary_labels, summary_values)
    axes[1].set_title("Mean compute time by iteration group")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(True, axis="y", alpha=0.2)

    if common_detail_generations and matched_slowdown:
        axes[2].plot(common_detail_generations, matched_slowdown, marker="o", color="#81B29A", linewidth=1.6)
        axes[2].axhline(1.0, color="#3D405B", linestyle="--", linewidth=1.0)
        axes[2].set_title(
            f"Printing slowdown on tracked-detail generations (mean {(_mean(matched_slowdown) - 1.0) * 100.0:+.2f}%)"
        )
        axes[2].set_xlabel("Generation")
        axes[2].set_ylabel("printing / non-printing")
        axes[2].grid(True, alpha=0.2)
    else:
        axes[2].text(
            0.5,
            0.5,
            "No overlapping tracked-detail generations between the two saved runs.",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_axis_off()

    figure.tight_layout()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_OVERVIEW_PLOT)
    plt.close(figure)

    comparable_sections = sorted(
        {
            section_name
            for row in baseline_detail_rows + printing_detail_rows
            for section_name in row["sections"].keys()
        }
    )
    baseline_section_means = _mean_section_seconds(baseline_detail_rows, comparable_sections)
    printing_section_means = _mean_section_seconds(printing_detail_rows, comparable_sections)

    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 16), dpi=180)
    x_positions = list(range(len(comparable_sections)))
    baseline_bars = [baseline_section_means.get(name) or 0.0 for name in comparable_sections]
    printing_bars = [printing_section_means.get(name) or 0.0 for name in comparable_sections]
    axes[0].bar([x - 0.2 for x in x_positions], baseline_bars, width=0.4, color="#3D405B", label="non-printing")
    axes[0].bar([x + 0.2 for x in x_positions], printing_bars, width=0.4, color="#E07A5F", label="printing")
    axes[0].set_title("Mean profiled section time on overlapping tracked-detail generations")
    axes[0].set_ylabel("Seconds")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(comparable_sections, rotation=35, ha="right")
    axes[0].grid(True, axis="y", alpha=0.2)
    axes[0].legend()

    comparable_ratio_sections = []
    comparable_ratios = []
    for section_name in comparable_sections:
        ratio = _safe_ratio(printing_section_means.get(section_name), baseline_section_means.get(section_name))
        if ratio is not None:
            comparable_ratio_sections.append(section_name)
            comparable_ratios.append(ratio)
    axes[1].bar(comparable_ratio_sections, comparable_ratios, color="#81B29A")
    axes[1].axhline(1.0, color="#3D405B", linestyle="--", linewidth=1.0)
    axes[1].set_title("Section slowdown ratio (printing / non-printing)")
    axes[1].set_ylabel("Ratio")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, axis="y", alpha=0.2)

    for section_name, color, label in (
        ("training_phase", "#B30000", "printing training_phase"),
        ("training_phase_rollout", "#E07A5F", "printing training_phase_rollout"),
    ):
        generations, values = _phase_series(printing_detail_rows, section_name)
        axes[2].plot(generations, values, marker="o", linewidth=1.6, color=color, label=label)
    for section_name, color, label in (
        ("training_phase", "#1D3557", "non-printing training_phase"),
        ("training_phase_rollout", "#3D405B", "non-printing training_phase_rollout"),
    ):
        generations, values = _phase_series(baseline_detail_rows, section_name)
        axes[2].plot(generations, values, marker="o", linewidth=1.6, color=color, label=label)
    axes[2].set_title("Tracked-detail phase timings by generation")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_PHASES_PLOT)
    plt.close(figure)

    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12), dpi=180)
    axes = axes.reshape(-1)

    baseline_active_ticks = [row.get("training_phase_stats", {}).get("active_individual_ticks") for row in baseline_detail_rows]
    printing_active_ticks = [row.get("training_phase_stats", {}).get("active_individual_ticks") for row in printing_detail_rows]
    baseline_rollout_seconds = [row["sections"].get("training_phase_rollout") for row in baseline_detail_rows]
    printing_rollout_seconds = [row["sections"].get("training_phase_rollout") for row in printing_detail_rows]
    axes[0].scatter(baseline_active_ticks, baseline_rollout_seconds, color="#3D405B", label="non-printing")
    axes[0].scatter(printing_active_ticks, printing_rollout_seconds, color="#E07A5F", label="printing")
    axes[0].set_title("Training rollout time vs simulated individual ticks")
    axes[0].set_xlabel("Active individual ticks")
    axes[0].set_ylabel("training_phase_rollout seconds")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    baseline_seconds_per_tick = [
        _safe_ratio(row["sections"].get("training_phase_rollout"), row.get("training_phase_stats", {}).get("active_individual_ticks"))
        for row in baseline_detail_rows
    ]
    printing_seconds_per_tick = [
        _safe_ratio(row["sections"].get("training_phase_rollout"), row.get("training_phase_stats", {}).get("active_individual_ticks"))
        for row in printing_detail_rows
    ]
    axes[1].plot(common_detail_generations, baseline_seconds_per_tick, marker="o", color="#3D405B", label="non-printing")
    axes[1].plot(common_detail_generations, printing_seconds_per_tick, marker="o", color="#E07A5F", label="printing")
    axes[1].set_title("Rollout seconds per simulated individual tick")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Seconds per active individual tick")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend()

    baseline_mean_run_ticks = [row.get("training_phase_stats", {}).get("mean_completed_run_ticks") for row in baseline_detail_rows]
    printing_mean_run_ticks = [row.get("training_phase_stats", {}).get("mean_completed_run_ticks") for row in printing_detail_rows]
    axes[2].plot(common_detail_generations, baseline_mean_run_ticks, marker="o", color="#3D405B", label="non-printing")
    axes[2].plot(common_detail_generations, printing_mean_run_ticks, marker="o", color="#E07A5F", label="printing")
    axes[2].set_title("Mean ticks used per completed maze run")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Ticks")
    axes[2].grid(True, alpha=0.2)
    axes[2].legend()

    baseline_full_length_fraction = [row.get("training_phase_stats", {}).get("full_length_run_fraction") for row in baseline_detail_rows]
    printing_full_length_fraction = [row.get("training_phase_stats", {}).get("full_length_run_fraction") for row in printing_detail_rows]
    axes[3].plot(common_detail_generations, baseline_full_length_fraction, marker="o", color="#3D405B", label="non-printing")
    axes[3].plot(common_detail_generations, printing_full_length_fraction, marker="o", color="#E07A5F", label="printing")
    axes[3].set_title("Fraction of full 7-tick maze runs")
    axes[3].set_xlabel("Generation")
    axes[3].set_ylabel("Fraction")
    axes[3].grid(True, alpha=0.2)
    axes[3].legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_TRAINING_PLOT)
    plt.close(figure)

    print(f"Saved overview comparison plot to: {OUTPUT_OVERVIEW_PLOT}")
    print(f"Saved phase comparison plot to: {OUTPUT_PHASES_PLOT}")
    print(f"Saved training-detail comparison plot to: {OUTPUT_TRAINING_PLOT}")

    if metadata_differences:
        print("Comparison note: these profiles were produced with different settings, so behavior-driven timing differences may dominate:")
        for key, baseline_value, printing_value in metadata_differences:
            print(f"  {key}: non-printing={baseline_value!r}, printing={printing_value!r}")

    if comparable_ratio_sections:
        print("Mean section slowdown ratios (printing / non-printing):")
        for section_name, ratio in zip(comparable_ratio_sections, comparable_ratios):
            print(f"  {section_name}: {ratio:.4f}x")
    mean_baseline_seconds_per_tick = _mean([value for value in baseline_seconds_per_tick if value is not None])
    mean_printing_seconds_per_tick = _mean([value for value in printing_seconds_per_tick if value is not None])
    if mean_baseline_seconds_per_tick is not None and mean_printing_seconds_per_tick is not None:
        print("Mean training rollout cost per active individual tick:")
        print(f"  non-printing: {mean_baseline_seconds_per_tick:.8f}s")
        print(f"  printing:     {mean_printing_seconds_per_tick:.8f}s")
        print(
            f"  slowdown:     {mean_printing_seconds_per_tick / mean_baseline_seconds_per_tick:.4f}x"
        )


if __name__ == "__main__":
    main()
