"""Instrumented objective with terminal tracking plus plot-ready history capture."""

import torch

from sim_core.constants import N
from sim_core.fitness_terms import compute_l1_penalty
from sim_core.genome_codec import unflatten_genome
from sim_core.maze_task import simulate_training_phase
from sim_core.maze_task_printing import simulate_training_phase_printing
from sim_core.replay_task import assign_replay_reward, simulate_replay_phase

# ==== 1) CONSTANTS ==============================================================
REPLAY_REWARD_METHOD = "zero"
TRAINING_CONTEXT_IS_A = True

_EVALUATION_CALL_COUNT = 0
_TOTAL_GENERATIONS = None
_PRINT_INTERVAL = None
_MAX_NETWORKS_PREVIEW = None
_MAX_RUNS_PREVIEW = None
_HIST_BIN_WIDTH = None

_TRACKED_GENERATIONS = []
_TRACKED_RECORDS = []
_REWARD_EVOLUTION = {
    "generation": [],
    "mean_eval": [],
    "median_eval": [],
    "pop_best_eval": [],
}


# ==== 2) PUBLIC CONTROL + HISTORY ACCESS =======================================
def configure_printing(
    total_generations,
    print_interval,
    max_networks_preview,
    max_runs_preview,
    hist_bin_width,
):
    """Set printing cadence and clear history buffers for a fresh run."""
    global _TOTAL_GENERATIONS, _PRINT_INTERVAL
    global _MAX_NETWORKS_PREVIEW, _MAX_RUNS_PREVIEW, _HIST_BIN_WIDTH
    global _EVALUATION_CALL_COUNT, _TRACKED_GENERATIONS, _TRACKED_RECORDS, _REWARD_EVOLUTION

    _TOTAL_GENERATIONS = total_generations
    _PRINT_INTERVAL = print_interval
    _MAX_NETWORKS_PREVIEW = max_networks_preview
    _MAX_RUNS_PREVIEW = max_runs_preview
    _HIST_BIN_WIDTH = hist_bin_width

    _EVALUATION_CALL_COUNT = 0
    _TRACKED_GENERATIONS = []
    _TRACKED_RECORDS = []
    _REWARD_EVOLUTION = {
        "generation": [],
        "mean_eval": [],
        "median_eval": [],
        "pop_best_eval": [],
    }


def get_printing_history():
    """Return tracked generations, detailed snapshots, and all-generation reward stats."""
    return {
        "tracked_generations": _TRACKED_GENERATIONS,
        "tracked_records": _TRACKED_RECORDS,
        "reward_evolution": _REWARD_EVOLUTION,
    }


# ==== 3) TERMINAL-FORMATTING HELPERS ===========================================
def _format_table(headers, rows):
    cols = len(headers)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx in range(cols):
            widths[idx] = max(widths[idx], len(str(row[idx])))

    def _line(char, cross):
        return cross + cross.join(char * (w + 2) for w in widths) + cross

    out = [_line("-", "+")]
    header_row = "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(cols)) + " |"
    out.append(header_row)
    out.append(_line("=", "+"))
    for row in rows:
        out.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(cols)) + " |")
    out.append(_line("-", "+"))
    return "\n".join(out)


def _hist_bin_edges(values, bin_width):
    min_v = float(torch.min(values).item())
    max_v = float(torch.max(values).item())
    start = bin_width * torch.floor(torch.tensor(min_v / bin_width)).item()
    end = bin_width * torch.ceil(torch.tensor(max_v / bin_width)).item()
    bins = max(1, int(round((end - start) / bin_width)))
    return start, end, bins


def _ascii_hist(values, bin_width, width):
    values = values.float()
    min_v, max_v, bins = _hist_bin_edges(values, bin_width)
    if min_v == max_v:
        return f"All values are {min_v:.4f}"

    hist = torch.histc(values, bins=bins, min=min_v, max=max_v)
    max_count = float(torch.max(hist).item())
    lines = []
    for i in range(bins):
        left = min_v + (max_v - min_v) * (i / bins)
        right = min_v + (max_v - min_v) * ((i + 1) / bins)
        count = int(hist[i].item())
        bar_len = int((count / max_count) * width)
        bar = "#" * bar_len
        lines.append(f"{left:8.3f}..{right:8.3f} | {bar} ({count})")
    return "\n".join(lines)


def _decision_symbol(decision, crashed, rewarded, big_reward):
    if crashed:
        return "x" if decision == -1 else "X"
    if decision == -1:
        return "."
    if rewarded:
        if big_reward:
            return "L" if decision == 0 else "R"
        return "l" if decision == 0 else "r"
    return "."


def _should_print(evaluation_idx):
    if evaluation_idx == 1:
        return True
    if evaluation_idx == _TOTAL_GENERATIONS:
        return True
    return evaluation_idx % _PRINT_INTERVAL == 0


# ==== 4) TRACKING SNAPSHOT PRINT =================================================
def _print_tracking_block(
    evaluation_idx,
    pop,
    training_reward_cpu,
    unregularized_reward_cpu,
    complexity_cpu,
    l1_penalty_cpu,
    regularized_fitness_cpu,
    frob_start_cpu,
    frob_end_cpu,
    frob_delta_cpu,
    tracking,
):
    decisions = tracking["decisions_by_run"].detach().cpu()
    crashed = tracking["crashed_by_run"].detach().cpu()
    rewarded = tracking["rewarded_by_run"].detach().cpu()
    big_reward = tracking["big_reward_by_run"].detach().cpu()

    reward_hist = _ascii_hist(regularized_fitness_cpu, _HIST_BIN_WIDTH, 36)

    start_mean = float(frob_start_cpu.mean().item())
    end_mean = float(frob_end_cpu.mean().item())
    delta_mean = float(frob_delta_cpu.mean().item())
    frob_table = _format_table(
        ["Metric", "Mean", "Min", "Max"],
        [
            ("Frobenius start", f"{start_mean:.4f}", f"{float(frob_start_cpu.min().item()):.4f}", f"{float(frob_start_cpu.max().item()):.4f}"),
            ("Frobenius end", f"{end_mean:.4f}", f"{float(frob_end_cpu.min().item()):.4f}", f"{float(frob_end_cpu.max().item()):.4f}"),
            ("Delta end-start", f"{delta_mean:.4f}", f"{float(frob_delta_cpu.min().item()):.4f}", f"{float(frob_delta_cpu.max().item()):.4f}"),
        ],
    )

    total_turns = int((decisions != -1).sum().item())
    left_turns = int((decisions == 0).sum().item())
    right_turns = int((decisions == 1).sum().item())
    crash_count = int(crashed.sum().item())
    reward_count = int(rewarded.sum().item())
    decisions_table = _format_table(
        ["Decision metric", "Count"],
        [
            ("Turns recorded", total_turns),
            ("Left turns", left_turns),
            ("Right turns", right_turns),
            ("Crashes", crash_count),
            ("Rewarded terminations", reward_count),
        ],
    )

    preview_networks = min(pop, _MAX_NETWORKS_PREVIEW)
    preview_runs = min(decisions.shape[1], _MAX_RUNS_PREVIEW)
    preview_rows = []
    for net_idx in range(preview_networks):
        seq = []
        for run_idx in range(preview_runs):
            seq.append(_decision_symbol(
                int(decisions[net_idx, run_idx].item()),
                bool(crashed[net_idx, run_idx].item()),
                bool(rewarded[net_idx, run_idx].item()),
                bool(big_reward[net_idx, run_idx].item()),
            ))
        preview_rows.append((f"net_{net_idx}", " ".join(seq)))
    decision_preview_table = _format_table(["Network", f"First {preview_runs} runs"], preview_rows)

    print()
    print("=" * 90)
    print(f"TRACKING SNAPSHOT | generation {evaluation_idx}/{_TOTAL_GENERATIONS}")
    print("=" * 90)
    print(f"Population size: {pop}")
    print()
    print(_format_table(
        ["Reward metric", "Mean", "Min", "Max"],
        [
            ("Training reward", f"{float(training_reward_cpu.mean().item()):.4f}", f"{float(training_reward_cpu.min().item()):.4f}", f"{float(training_reward_cpu.max().item()):.4f}"),
            ("Reward before L1", f"{float(unregularized_reward_cpu.mean().item()):.4f}", f"{float(unregularized_reward_cpu.min().item()):.4f}", f"{float(unregularized_reward_cpu.max().item()):.4f}"),
            ("L1 complexity", f"{float(complexity_cpu.mean().item()):.4f}", f"{float(complexity_cpu.min().item()):.4f}", f"{float(complexity_cpu.max().item()):.4f}"),
            ("L1 penalty", f"{float(l1_penalty_cpu.mean().item()):.4f}", f"{float(l1_penalty_cpu.min().item()):.4f}", f"{float(l1_penalty_cpu.max().item()):.4f}"),
            ("Fitness after L1", f"{float(regularized_fitness_cpu.mean().item()):.4f}", f"{float(regularized_fitness_cpu.min().item()):.4f}", f"{float(regularized_fitness_cpu.max().item()):.4f}"),
        ],
    ))
    print()
    print("Reward histogram:")
    print(reward_hist)
    print()
    print("Weight norm summary:")
    print(frob_table)
    print()
    print("Decision summary:")
    print(decisions_table)
    print()
    print("Decision preview legend: x=crash before turn, X=crash after turn, l/r=small reward, L/R=big reward, .=no event")
    print(decision_preview_table)
    print("=" * 90)
    print()


# ==== 5) HISTORY WRITER =========================================================
def _record_history(
    evaluation_idx,
    regularized_fitness,
    frob_start_cpu,
    frob_end_cpu,
    tracking,
):
    fit_cpu = regularized_fitness.detach().cpu()
    _REWARD_EVOLUTION["generation"].append(evaluation_idx)
    _REWARD_EVOLUTION["mean_eval"].append(float(fit_cpu.mean().item()))
    _REWARD_EVOLUTION["median_eval"].append(float(fit_cpu.median().item()))
    _REWARD_EVOLUTION["pop_best_eval"].append(float(fit_cpu.max().item()))

    if tracking is None:
        return

    _TRACKED_GENERATIONS.append(evaluation_idx)
    _TRACKED_RECORDS.append(
        {
            "generation": evaluation_idx,
            "fitness": fit_cpu,
            "frob_start": frob_start_cpu.clone(),
            "frob_end": frob_end_cpu.clone(),
            "decisions_by_run": tracking["decisions_by_run"].detach().cpu().clone(),
            "crashed_by_run": tracking["crashed_by_run"].detach().cpu().clone(),
            "rewarded_by_run": tracking["rewarded_by_run"].detach().cpu().clone(),
            "big_reward_by_run": tracking["big_reward_by_run"].detach().cpu().clone(),
            "sensory_cue_by_run": tracking["sensory_cue_by_run"].detach().cpu().clone(),
        }
    )


# ==== 6) FITNESS EVALUATION =====================================================
def evaluate_generation_printing(genome_flat, device, noise_generator, reward_generator, l1_lambda):
    global _EVALUATION_CALL_COUNT
    _EVALUATION_CALL_COUNT += 1
    evaluation_idx = _EVALUATION_CALL_COUNT
    should_print = _should_print(evaluation_idx)

    genome_flat = genome_flat.clone()
    pop = genome_flat.shape[0]
    genome = unflatten_genome(genome_flat, pop)

    state0 = torch.zeros(pop, N, device=device)
    tracking = None
    if should_print:
        frob_start = torch.linalg.matrix_norm(genome["W"], ord="fro", dim=(1, 2))
        state, W_after_training, training_reward, tracking = simulate_training_phase_printing(
            state0, genome["W"], genome["M"], genome["A"], genome["B"], genome["C"], genome["D"],
            genome["beta"], genome["eta"], TRAINING_CONTEXT_IS_A,
            noise_generator, reward_generator, device,
        )
    else:
        state, W_after_training, training_reward = simulate_training_phase(
            state0, genome["W"], genome["M"], genome["A"], genome["B"], genome["C"], genome["D"],
            genome["beta"], genome["eta"], TRAINING_CONTEXT_IS_A,
            noise_generator, reward_generator, device,
        )
        frob_start = torch.linalg.matrix_norm(genome["W"], ord="fro", dim=(1, 2))

    _, W_after_replay, replay_trace = simulate_replay_phase(
        state, W_after_training, genome["M"], genome["A"], genome["B"], genome["C"], genome["D"],
        genome["beta"], genome["eta"], noise_generator, device,
    )

    replay_reward = assign_replay_reward(replay_trace, REPLAY_REWARD_METHOD)
    unregularized_reward = training_reward + replay_reward
    complexity, l1_penalty = compute_l1_penalty(genome_flat, l1_lambda)
    regularized_fitness = unregularized_reward - l1_penalty
    frob_end = torch.linalg.matrix_norm(W_after_replay, ord="fro", dim=(1, 2))
    frob_delta = frob_end - frob_start

    if should_print:
        _print_tracking_block(
            evaluation_idx,
            pop,
            training_reward.detach().cpu(),
            unregularized_reward.detach().cpu(),
            complexity.detach().cpu(),
            l1_penalty.detach().cpu(),
            regularized_fitness.detach().cpu(),
            frob_start.detach().cpu(),
            frob_end.detach().cpu(),
            frob_delta.detach().cpu(),
            tracking,
        )
        _record_history(
            evaluation_idx,
            regularized_fitness,
            frob_start.detach().cpu(),
            frob_end.detach().cpu(),
            tracking,
        )
    else:
        _record_history(
            evaluation_idx,
            regularized_fitness,
            frob_start.detach().cpu(),
            frob_end.detach().cpu(),
            None,
        )

    return regularized_fitness


def fitness_function_printing(genome_flat, device, noise_generator, reward_generator, l1_lambda):
    """Vectorized EvoTorch objective entrypoint."""
    return evaluate_generation_printing(genome_flat, device, noise_generator, reward_generator, l1_lambda)
