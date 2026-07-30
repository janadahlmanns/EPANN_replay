"""Instrumented objective function with gated, readable tracking output."""

import torch
from sim_core.constants import N
from sim_core.genome_codec import unflatten_genome
from sim_core.maze_task import simulate_training_phase
from sim_core.maze_task_printing import simulate_training_phase_printing
from sim_core.replay_task import simulate_replay_phase, assign_replay_reward

REPLAY_REWARD_METHOD = "zero"
TRAINING_CONTEXT_IS_A = True
_EVALUATION_CALL_COUNT = 0
_TOTAL_GENERATIONS = None
_PRINT_INTERVAL = 50
_MAX_NETWORKS_PREVIEW = 6
_MAX_RUNS_PREVIEW = 20
_HIST_BINS = 20


def configure_printing(
    total_generations=None,
    print_interval=50,
    max_networks_preview=6,
    max_runs_preview=20,
    hist_bins=20,
):
    """Configure logging cadence and verbosity for the printing objective."""
    global _TOTAL_GENERATIONS, _PRINT_INTERVAL
    global _MAX_NETWORKS_PREVIEW, _MAX_RUNS_PREVIEW, _HIST_BINS

    _TOTAL_GENERATIONS = total_generations
    _PRINT_INTERVAL = print_interval
    _MAX_NETWORKS_PREVIEW = max_networks_preview
    _MAX_RUNS_PREVIEW = max_runs_preview
    _HIST_BINS = hist_bins


def _format_table(headers, rows):
    cols = len(headers)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx in range(cols):
            widths[idx] = max(widths[idx], len(str(row[idx])))

    def _line(char="-", cross="+"):
        return cross + cross.join(char * (w + 2) for w in widths) + cross

    out = [_line()]
    header_row = "| " + " | ".join(str(headers[i]).ljust(widths[i]) for i in range(cols)) + " |"
    out.append(header_row)
    out.append(_line("=","+"))
    for row in rows:
        out.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(cols)) + " |")
    out.append(_line())
    return "\n".join(out)


def _ascii_hist(values, bins, width=36):
    values = values.float()
    min_v = float(torch.min(values).item())
    max_v = float(torch.max(values).item())
    if min_v == max_v:
        return f"All values are {min_v:.4f}"

    hist = torch.histc(values, bins=bins, min=min_v, max=max_v)
    max_count = float(torch.max(hist).item())
    lines = []
    for i in range(bins):
        left = min_v + (max_v - min_v) * (i / bins)
        right = min_v + (max_v - min_v) * ((i + 1) / bins)
        count = int(hist[i].item())
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
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
    if _TOTAL_GENERATIONS is None:
        return True
    if evaluation_idx == 1:
        return True
    if evaluation_idx == _TOTAL_GENERATIONS:
        return True
    return evaluation_idx % _PRINT_INTERVAL == 0


def _print_tracking_block(
    evaluation_idx,
    pop,
    training_reward_cpu,
    total_reward_cpu,
    frob_start_cpu,
    frob_end_cpu,
    frob_delta_cpu,
    tracking,
):
    decisions = tracking["decisions_by_run"].detach().cpu()
    crashed = tracking["crashed_by_run"].detach().cpu()
    rewarded = tracking["rewarded_by_run"].detach().cpu()
    big_reward = tracking["big_reward_by_run"].detach().cpu()

    reward_hist = _ascii_hist(total_reward_cpu, _HIST_BINS)

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
    if _TOTAL_GENERATIONS is None:
        print(f"TRACKING SNAPSHOT | evaluation {evaluation_idx}")
    else:
        print(f"TRACKING SNAPSHOT | generation {evaluation_idx}/{_TOTAL_GENERATIONS}")
    print("=" * 90)
    print(f"Population size: {pop}")
    print()
    print(_format_table(
        ["Reward metric", "Mean", "Min", "Max"],
        [
            ("Training reward", f"{float(training_reward_cpu.mean().item()):.4f}", f"{float(training_reward_cpu.min().item()):.4f}", f"{float(training_reward_cpu.max().item()):.4f}"),
            ("Total reward", f"{float(total_reward_cpu.mean().item()):.4f}", f"{float(total_reward_cpu.min().item()):.4f}", f"{float(total_reward_cpu.max().item()):.4f}"),
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


def evaluate_generation_printing(genome_flat, device, noise_generator, reward_generator):
    global _EVALUATION_CALL_COUNT
    _EVALUATION_CALL_COUNT += 1
    should_print = _should_print(_EVALUATION_CALL_COUNT)

    genome_flat = genome_flat.clone()
    pop = genome_flat.shape[0]
    genome = unflatten_genome(genome_flat, pop)

    state0 = torch.zeros(pop, N, device=device)
    tracking = None
    if should_print:
        initial_W = genome["W"]
        frob_start = torch.linalg.matrix_norm(initial_W, ord="fro", dim=(1, 2))
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

    _, W_after_replay, replay_trace = simulate_replay_phase(
        state, W_after_training, genome["M"], genome["A"], genome["B"], genome["C"], genome["D"],
        genome["beta"], genome["eta"], noise_generator, device,
    )
    replay_reward = assign_replay_reward(replay_trace, REPLAY_REWARD_METHOD)
    total_reward = training_reward + replay_reward

    if should_print:
        frob_end = torch.linalg.matrix_norm(W_after_replay, ord="fro", dim=(1, 2))
        frob_delta = frob_end - frob_start
        _print_tracking_block(
            evaluation_idx=_EVALUATION_CALL_COUNT,
            pop=pop,
            training_reward_cpu=training_reward.detach().cpu(),
            total_reward_cpu=total_reward.detach().cpu(),
            frob_start_cpu=frob_start.detach().cpu(),
            frob_end_cpu=frob_end.detach().cpu(),
            frob_delta_cpu=frob_delta.detach().cpu(),
            tracking=tracking,
        )

    return total_reward


def fitness_function_printing(genome_flat, device, noise_generator, reward_generator):
    return evaluate_generation_printing(genome_flat, device, noise_generator, reward_generator)
