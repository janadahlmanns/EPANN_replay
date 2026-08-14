"""Instrumented objective with terminal tracking plus plot-ready history capture."""

import torch

from sim_core import constants
from sim_core.fitness_terms import compute_l1_penalty
from sim_core.genome_codec import sample_initial_weights, unflatten_genome
from sim_core.maze_task import simulate_training_phase
from sim_core.paradigm import PHASE_REPLAY, PHASE_TRAIN_A, PHASE_TRAIN_B
from sim_core.replay_task import assign_replay_reward, simulate_replay_phase

# ==== 1) CONSTANTS ==============================================================
REPLAY_REWARD_METHOD = "zero"
PHASE_CONTEXT = {PHASE_TRAIN_A: "A", PHASE_TRAIN_B: "B"}

_TOTAL_GENERATIONS = None
_PRINT_INTERVAL = None
_MAX_NETWORKS_PREVIEW = None
_MAX_RUNS_PREVIEW = None
_CHAIN_LABEL = None  # printed as "CHAIN <label>" -- identifies which run_batch.py chain (or
                      # manual run) this process's terminal output belongs to
_CONFIG_NAME = None  # printed in place of a bare population-size number, so the terminal
                      # output says which config is actually running

_TRACKED_GENERATIONS = []
_TRACKED_RECORDS = []
_REWARD_EVOLUTION = {
    "generation": [],
    "mean_eval": [],
    "median_eval": [],
    "pop_best_eval": [],
    "std_eval": [],
    "training_reward_mean": [],
    "training_reward_median": [],
    "training_reward_best": [],
    "l1_penalty_mean": [],
    "l1_penalty_median": [],
    "l1_penalty_best": [],
}
_CUE_IMPORTANCE_HISTORY = {
    "generation": [],
    "context_importance_mean": [],
    "context_importance_min": [],
    "context_importance_max": [],
    "sensory_importance_mean": [],
    "sensory_importance_min": [],
    "sensory_importance_max": [],
}


# ==== 2) PUBLIC CONTROL + HISTORY ACCESS =======================================
def configure_printing(
    total_generations,
    print_interval,
    max_networks_preview,
    max_runs_preview,
    chain_label,
    config_name,
):
    """Set printing cadence and clear history buffers for a fresh run."""
    global _TOTAL_GENERATIONS, _PRINT_INTERVAL
    global _MAX_NETWORKS_PREVIEW, _MAX_RUNS_PREVIEW
    global _CHAIN_LABEL, _CONFIG_NAME
    global _TRACKED_GENERATIONS, _TRACKED_RECORDS, _REWARD_EVOLUTION, _CUE_IMPORTANCE_HISTORY

    _TOTAL_GENERATIONS = total_generations
    _PRINT_INTERVAL = print_interval
    _MAX_NETWORKS_PREVIEW = max_networks_preview
    _MAX_RUNS_PREVIEW = max_runs_preview
    _CHAIN_LABEL = chain_label
    _CONFIG_NAME = config_name

    _TRACKED_GENERATIONS = []
    _TRACKED_RECORDS = []
    _REWARD_EVOLUTION = {
        "generation": [],
        "mean_eval": [],
        "median_eval": [],
        "pop_best_eval": [],
        "std_eval": [],
        "training_reward_mean": [],
        "training_reward_median": [],
        "training_reward_best": [],
        "l1_penalty_mean": [],
        "l1_penalty_median": [],
        "l1_penalty_best": [],
    }
    _CUE_IMPORTANCE_HISTORY = {
        "generation": [],
        "context_importance_mean": [],
        "context_importance_min": [],
        "context_importance_max": [],
        "sensory_importance_mean": [],
        "sensory_importance_min": [],
        "sensory_importance_max": [],
    }


def get_printing_history():
    """Return tracked generations, detailed snapshots, and all-generation reward stats."""
    return {
        "tracked_generations": _TRACKED_GENERATIONS,
        "tracked_records": _TRACKED_RECORDS,
        "reward_evolution": _REWARD_EVOLUTION,
        "cue_importance_history": _CUE_IMPORTANCE_HISTORY,
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


def _hstack_tables(tables, spacer="    "):
    """Lay out multiple already-formatted (newline-joined) tables side by side as
    columns instead of stacked vertically, padding shorter tables with blank space
    so every row still lines up."""
    split_tables = [table.split("\n") for table in tables]
    widths = [len(lines[0]) for lines in split_tables]
    max_rows = max(len(lines) for lines in split_tables)
    rows = []
    for row_idx in range(max_rows):
        row_parts = [
            lines[row_idx] if row_idx < len(lines) else " " * width
            for lines, width in zip(split_tables, widths)
        ]
        rows.append(spacer.join(row_parts))
    return "\n".join(rows)


def _decision_symbol(decision, crashed, rewarded, correct_arm):
    """Map one run's outcome to one of the 9 event symbols (or '.' for no event).

    decision: -1 = never turned, 0 = left, 1 = right
    correct_arm: whether `decision` matches this run's big-reward arm
                 (only meaningful when decision != -1)
    """
    if decision == -1:
        return "x" if crashed else "."

    letter = ("L" if decision == 0 else "R") if correct_arm else ("l" if decision == 0 else "r")
    if crashed:
        return letter + "x"
    if rewarded:
        return letter
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
    tracking,
):
    decisions = tracking["decisions_by_run"].detach().cpu()
    crashed = tracking["crashed_by_run"].detach().cpu()
    rewarded = tracking["rewarded_by_run"].detach().cpu()
    correct_arm = tracking["correct_arm_by_run"].detach().cpu()

    reward_table = _format_table(
        ["Reward metric", "Mean", "Min", "Max"],
        [
            ("Training reward", f"{float(training_reward_cpu.mean().item()):.4f}", f"{float(training_reward_cpu.min().item()):.4f}", f"{float(training_reward_cpu.max().item()):.4f}"),
            ("Reward before L1", f"{float(unregularized_reward_cpu.mean().item()):.4f}", f"{float(unregularized_reward_cpu.min().item()):.4f}", f"{float(unregularized_reward_cpu.max().item()):.4f}"),
            ("L1 complexity", f"{float(complexity_cpu.mean().item()):.4f}", f"{float(complexity_cpu.min().item()):.4f}", f"{float(complexity_cpu.max().item()):.4f}"),
            ("L1 penalty", f"{float(l1_penalty_cpu.mean().item()):.4f}", f"{float(l1_penalty_cpu.min().item()):.4f}", f"{float(l1_penalty_cpu.max().item()):.4f}"),
            ("Fitness after L1", f"{float(regularized_fitness_cpu.mean().item()):.4f}", f"{float(regularized_fitness_cpu.min().item()):.4f}", f"{float(regularized_fitness_cpu.max().item()):.4f}"),
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
                bool(correct_arm[net_idx, run_idx].item()),
            ))
        preview_rows.append((f"net_{net_idx}", " ".join(seq)))
    decision_preview_table = _format_table(["Network", f"First {preview_runs} runs"], preview_rows)

    print()
    print("=" * 90)
    print(f"CHAIN {_CHAIN_LABEL} | generation {evaluation_idx}/{_TOTAL_GENERATIONS}")
    print("=" * 90)
    print(f"Config: {_CONFIG_NAME}")
    print()
    print(_hstack_tables([reward_table, decisions_table, decision_preview_table]))
    print("=" * 90)
    print()


# ==== 5) HISTORY WRITER =========================================================
def _record_history(
    evaluation_idx,
    regularized_fitness,
    training_reward,
    l1_penalty,
    frob_start_cpu,
    frob_end_cpu,
    weights_start_cpu,
    weights_end_cpu,
    tracking,
):
    fit_cpu = regularized_fitness.detach().cpu()
    tr_cpu = training_reward.detach().cpu()
    l1_cpu = l1_penalty.detach().cpu()
    _REWARD_EVOLUTION["generation"].append(evaluation_idx)
    _REWARD_EVOLUTION["mean_eval"].append(float(fit_cpu.mean().item()))
    _REWARD_EVOLUTION["median_eval"].append(float(fit_cpu.median().item()))
    _REWARD_EVOLUTION["pop_best_eval"].append(float(fit_cpu.max().item()))
    _REWARD_EVOLUTION["std_eval"].append(float(fit_cpu.std(unbiased=False).item()))
    _REWARD_EVOLUTION["training_reward_mean"].append(float(tr_cpu.mean().item()))
    _REWARD_EVOLUTION["training_reward_median"].append(float(tr_cpu.median().item()))
    _REWARD_EVOLUTION["training_reward_best"].append(float(tr_cpu.max().item()))
    _REWARD_EVOLUTION["l1_penalty_mean"].append(float(l1_cpu.mean().item()))
    _REWARD_EVOLUTION["l1_penalty_median"].append(float(l1_cpu.median().item()))
    _REWARD_EVOLUTION["l1_penalty_best"].append(float(l1_cpu.max().item()))

    if tracking is None:
        return

    _TRACKED_GENERATIONS.append(evaluation_idx)
    _TRACKED_RECORDS.append(
        {
            "generation": evaluation_idx,
            "fitness": fit_cpu,
            "frob_start": frob_start_cpu.clone(),
            "frob_end": frob_end_cpu.clone(),
            "weights_start": weights_start_cpu.clone(),
            "weights_end": weights_end_cpu.clone(),
            "decisions_by_run": tracking["decisions_by_run"].detach().cpu().clone(),
            "crashed_by_run": tracking["crashed_by_run"].detach().cpu().clone(),
            "rewarded_by_run": tracking["rewarded_by_run"].detach().cpu().clone(),
            "big_reward_by_run": tracking["big_reward_by_run"].detach().cpu().clone(),
            "sensory_cue_by_run": tracking["sensory_cue_by_run"].detach().cpu().clone(),
            "correct_arm_by_run": tracking["correct_arm_by_run"].detach().cpu().clone(),
        }
    )


def _record_cue_importance(evaluation_idx, context_importance, sensory_importance):
    """Store one tracked generation's context/sensory cue importance distribution."""
    context_cpu = context_importance.detach().cpu()
    sensory_cpu = sensory_importance.detach().cpu()
    _CUE_IMPORTANCE_HISTORY["generation"].append(evaluation_idx)
    _CUE_IMPORTANCE_HISTORY["context_importance_mean"].append(float(context_cpu.mean().item()))
    _CUE_IMPORTANCE_HISTORY["context_importance_min"].append(float(context_cpu.min().item()))
    _CUE_IMPORTANCE_HISTORY["context_importance_max"].append(float(context_cpu.max().item()))
    _CUE_IMPORTANCE_HISTORY["sensory_importance_mean"].append(float(sensory_cpu.mean().item()))
    _CUE_IMPORTANCE_HISTORY["sensory_importance_min"].append(float(sensory_cpu.min().item()))
    _CUE_IMPORTANCE_HISTORY["sensory_importance_max"].append(float(sensory_cpu.max().item()))


# ==== 6) PARADIGM EXECUTION =====================================================
def _concat_tracking_segments(segments):
    """Concatenate per-training-phase tracking dicts along the run axis (dim=1),
    so multiple trainA/trainB phases appear as one chronological run sequence."""
    return {key: torch.cat([segment[key] for segment in segments], dim=1) for key in segments[0]}


def _run_paradigm(genome, paradigm_phases, device, noise_generator, reward_generator, collect_tracking,
                   context_cues_on, sensory_cues_on, state, W):
    """Runs every (phase_type, value) in paradigm_phases in order, starting from the
    given (state, W) and chaining CTRNN state/weights across phases. Returns the
    final (state, W) plus the summed training and replay reward. Never mutates the
    state/W tensors passed in -- simulate_training_phase/simulate_replay_phase both
    clone-or-recompute rather than mutate in place -- so the same (state, W) can
    safely be reused as a starting checkpoint across multiple independent calls."""
    training_reward = torch.zeros(state.shape[0], device=device)
    replay_reward = torch.zeros(state.shape[0], device=device)
    tracking_segments = []

    for phase_type, value in paradigm_phases:
        if phase_type in PHASE_CONTEXT:
            result = simulate_training_phase(
                state, W, genome["M"], genome["A"], genome["B"], genome["C"], genome["D"],
                genome["beta"], genome["eta"], PHASE_CONTEXT[phase_type], value,
                context_cues_on, sensory_cues_on,
                noise_generator, reward_generator, device,
                collect_tracking=collect_tracking,
            )
            if collect_tracking:
                state, W, phase_reward, phase_tracking = result
                tracking_segments.append(phase_tracking)
            else:
                state, W, phase_reward = result
            training_reward = training_reward + phase_reward
        elif phase_type == PHASE_REPLAY:
            state, W, replay_trace = simulate_replay_phase(
                state, W, genome["M"], genome["A"], genome["B"], genome["C"], genome["D"],
                genome["beta"], genome["eta"], value, noise_generator, device,
            )
            replay_reward = replay_reward + assign_replay_reward(replay_trace, REPLAY_REWARD_METHOD)
        else:
            raise ValueError(f"Unknown phase type '{phase_type}' in paradigm.")

    tracking = _concat_tracking_segments(tracking_segments) if collect_tracking else None
    return state, W, training_reward, replay_reward, tracking


def _measure_cue_importance(genome, paradigm_phases, device, test_generator,
                             context_cues_on, sensory_cues_on, state_checkpoint, W_checkpoint):
    """Ablation importance: reward lost when one cue channel is clipped to zero,
    measured by continuing the SAME paradigm shape once more from the real
    post-evaluation (state_checkpoint, W_checkpoint) -- i.e. probing what the
    actually-trained network does, not a network retrained from scratch.

    Uses test_generator exclusively (never the evolutionary noise/reward
    generators), so this diagnostic draws no randomness from and has zero effect
    on the main evolutionary RNG stream. All three probes (baseline + two
    ablations) are replayed from the identical test_generator state, so they see
    the same maze draws/noise and differ only in which cue is ablated. Every
    probe's resulting state/W is discarded once its reward is read out -- nothing
    from testing carries out into the real evaluation, only the checkpoint carries in.
    """
    checkpoint_rng_state = test_generator.get_state()

    def _probe(probe_context_cues_on, probe_sensory_cues_on):
        test_generator.set_state(checkpoint_rng_state)
        _, _, probe_training_reward, probe_replay_reward, _ = _run_paradigm(
            genome, paradigm_phases, device, test_generator, test_generator, False,
            probe_context_cues_on, probe_sensory_cues_on, state_checkpoint, W_checkpoint,
        )
        return probe_training_reward + probe_replay_reward

    baseline_reward = _probe(context_cues_on, sensory_cues_on)
    context_ablated_reward = _probe(False, sensory_cues_on)
    sensory_ablated_reward = _probe(context_cues_on, False)

    context_importance = baseline_reward - context_ablated_reward
    sensory_importance = baseline_reward - sensory_ablated_reward
    return context_importance, sensory_importance


# ==== 7) FITNESS EVALUATION =====================================================
def evaluate_generation(genome_flat, device, noise_generator, reward_generator, test_generator,
                         weight_init_generator, l1_lambda, context_cues_on, sensory_cues_on,
                         evo_plasticity_on, paradigm_phases):
    evaluation_idx = len(_REWARD_EVOLUTION["generation"]) + 1
    should_print = _should_print(evaluation_idx)

    genome_flat = genome_flat.clone()
    pop = genome_flat.shape[0]
    genome = unflatten_genome(genome_flat, pop)
    if not evo_plasticity_on:
        genome["eta"] = torch.zeros_like(genome["eta"])

    # fresh initial weights every lifetime -- NOT read from the genome, see genome_codec.py
    W_init = sample_initial_weights(pop, device, weight_init_generator)

    frob_start = torch.linalg.matrix_norm(W_init, ord="fro", dim=(1, 2))
    state0 = torch.zeros(pop, constants.N, device=device)
    state_final, W_final, training_reward, replay_reward, tracking = _run_paradigm(
        genome, paradigm_phases, device, noise_generator, reward_generator, should_print,
        context_cues_on, sensory_cues_on, state0, W_init,
    )

    unregularized_reward = training_reward + replay_reward
    complexity, l1_penalty = compute_l1_penalty(genome_flat, l1_lambda)
    regularized_fitness = unregularized_reward - l1_penalty
    frob_end = torch.linalg.matrix_norm(W_final, ord="fro", dim=(1, 2))

    if should_print:
        _print_tracking_block(
            evaluation_idx,
            pop,
            training_reward.detach().cpu(),
            unregularized_reward.detach().cpu(),
            complexity.detach().cpu(),
            l1_penalty.detach().cpu(),
            regularized_fitness.detach().cpu(),
            tracking,
        )
        _record_history(
            evaluation_idx,
            regularized_fitness,
            training_reward,
            l1_penalty,
            frob_start.detach().cpu(),
            frob_end.detach().cpu(),
            W_init.detach().cpu(),
            W_final.detach().cpu(),
            tracking,
        )
        context_importance, sensory_importance = _measure_cue_importance(
            genome, paradigm_phases, device, test_generator,
            context_cues_on, sensory_cues_on, state_final, W_final,
        )
        _record_cue_importance(evaluation_idx, context_importance, sensory_importance)
    else:
        fit_cpu = regularized_fitness.detach().cpu()
        print(f"CHAIN {_CHAIN_LABEL} - iter {evaluation_idx} - best: {float(fit_cpu.max()):.1f} - median: {float(fit_cpu.median()):.1f}")
        _record_history(
            evaluation_idx,
            regularized_fitness,
            training_reward,
            l1_penalty,
            frob_start.detach().cpu(),
            frob_end.detach().cpu(),
            None,
            None,
            None,
        )

    return regularized_fitness


def fitness_function(genome_flat, device, noise_generator, reward_generator, test_generator,
                      weight_init_generator, l1_lambda, context_cues_on, sensory_cues_on,
                      evo_plasticity_on, paradigm_phases):
    """Vectorized EvoTorch objective entrypoint."""
    return evaluate_generation(genome_flat, device, noise_generator, reward_generator, test_generator,
                                weight_init_generator, l1_lambda, context_cues_on, sensory_cues_on,
                                evo_plasticity_on, paradigm_phases)