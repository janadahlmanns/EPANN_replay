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
_FORCE_TRACK_FIRST_N = None  # always track/print the first N evaluations of a run, regardless of
                              # _PRINT_INTERVAL -- general mechanism, but N is an es_method-specific
                              # VALUE chosen by run_evolution.py (see configure_printing's docstring)
_MAX_NETWORKS_PREVIEW = None
_MAX_RUNS_PREVIEW = None
_CHAIN_LABEL = None  # printed as "CHAIN <label>" -- identifies which run_batch.py chain (or
                      # manual run) this process's terminal output belongs to
_CONFIG_NAME = None  # printed in place of a bare population-size number, so the terminal
                      # output says which config is actually running
_EARLY_TERMINATION_ENABLED = None  # from run_evolution.py's CLI (early_termination_enabled) --
                                    # when False, _check_event_count_termination still detects and
                                    # prints a met criterion every time, but never sets
                                    # _TERMINATE_EARLY, so the run always goes the full configured
                                    # generation count regardless of a seemingly-collapsed population

_TERMINATE_EARLY = False  # set by _check_event_count_termination on a detailed-print generation
                           # when, for EITHER task (trainA/trainB), across the WHOLE tracked
                           # population, the search has collapsed onto a degenerate policy (see
                           # that function for the exact conditions) -- wasted compute that isn't
                           # going to develop into a viable solution. run_evolution.py's main loop
                           # checks should_terminate_early() and stops (saving everything
                           # normally, as if this were the last generation). Never set at all when
                           # _EARLY_TERMINATION_ENABLED is False -- see that flag's comment.

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
    "reward_importance_mean": [],
    "reward_importance_min": [],
    "reward_importance_max": [],
}
_TRANSFER_METRICS_HISTORY = {
    "generation": [],
    "fwt_mean": [],
    "fwt_min": [],
    "fwt_max": [],
    "bwt_mean": [],
    "bwt_min": [],
    "bwt_max": [],
    # Per-component traceability columns (R_<label>_own, baseline_<label>, etc. -- see
    # _measure_transfer_metrics's docstring) are NOT listed here: this project's task
    # count T is read from the paradigm, so the component set's size/names vary with T
    # (2 tasks -> 4 components, 3 tasks -> 7, ...). _record_transfer_metrics adds them
    # to this dict the first time it sees them (generation 1, always tracked -- see
    # _should_print), so every tracked generation ends up with the same key set.
}


# ==== 2) PUBLIC CONTROL + HISTORY ACCESS =======================================
def configure_printing(
    total_generations,
    print_interval,
    max_networks_preview,
    max_runs_preview,
    chain_label,
    config_name,
    early_termination_enabled,
    force_track_first_n,
):
    """Set printing cadence and clear history buffers for a fresh run.

    force_track_first_n: always track/print evaluation_idx 1..force_track_first_n,
    regardless of print_interval -- general mechanism (see _should_print), but the VALUE
    is es_method-specific, chosen by run_evolution.py. PGPE (and any method where one
    searcher.step() == exactly one real generation) wants 1, matching this project's
    original "always track generation 1" behavior exactly. Cosyne wants 2: its searcher's
    very first .step() secretly evaluates the raw initial population once before its
    first real generation (see sim_core/fitness.py's discard_earliest_recorded_generation
    docstring) -- that phantom evaluation is evaluation_idx 1 and always gets discarded,
    so evaluation_idx 2 (the first REAL generation, renumbered down to 1 afterward) must
    ALSO be force-tracked here, or it silently ends up with no tracking data at all
    whenever print_interval doesn't happen to divide 2 -- and tracking data cannot be
    reconstructed after the fact once a generation has already run without it."""
    global _TOTAL_GENERATIONS, _PRINT_INTERVAL, _FORCE_TRACK_FIRST_N
    global _MAX_NETWORKS_PREVIEW, _MAX_RUNS_PREVIEW
    global _CHAIN_LABEL, _CONFIG_NAME, _EARLY_TERMINATION_ENABLED, _TERMINATE_EARLY
    global _TRACKED_GENERATIONS, _TRACKED_RECORDS, _REWARD_EVOLUTION
    global _CUE_IMPORTANCE_HISTORY, _TRANSFER_METRICS_HISTORY

    _TOTAL_GENERATIONS = total_generations
    _PRINT_INTERVAL = print_interval
    _FORCE_TRACK_FIRST_N = force_track_first_n
    _MAX_NETWORKS_PREVIEW = max_networks_preview
    _MAX_RUNS_PREVIEW = max_runs_preview
    _CHAIN_LABEL = chain_label
    _CONFIG_NAME = config_name
    _EARLY_TERMINATION_ENABLED = early_termination_enabled
    _TERMINATE_EARLY = False

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
        "reward_importance_mean": [],
        "reward_importance_min": [],
        "reward_importance_max": [],
    }
    _TRANSFER_METRICS_HISTORY = {
        "generation": [],
        "fwt_mean": [],
        "fwt_min": [],
        "fwt_max": [],
        "bwt_mean": [],
        "bwt_min": [],
        "bwt_max": [],
        # per-component traceability columns added dynamically -- see module-level
        # _TRANSFER_METRICS_HISTORY's comment and _record_transfer_metrics
    }


def get_printing_history():
    """Return tracked generations, detailed snapshots, and all-generation reward stats."""
    return {
        "tracked_generations": _TRACKED_GENERATIONS,
        "tracked_records": _TRACKED_RECORDS,
        "reward_evolution": _REWARD_EVOLUTION,
        "cue_importance_history": _CUE_IMPORTANCE_HISTORY,
        "transfer_metrics_history": _TRANSFER_METRICS_HISTORY,
    }


def should_terminate_early():
    """True once _check_event_count_termination has flagged a degenerate (collapsed)
    population for some task. The caller (run_evolution.py's main loop) is responsible
    for actually stopping -- this module only ever sets the flag, never breaks
    anything itself."""
    return _TERMINATE_EARLY


def discard_earliest_recorded_generation():
    """Drop the single OLDEST recorded generation's entries from every history buffer
    (reward_evolution always; tracked_generations/tracked_records, cue_importance, and
    transfer_metrics too, each only if their own oldest entry is actually that same
    generation -- being tracked at all is conditional, see _should_print), THEN
    renumber every remaining entry's stored "generation" label down by 1.

    Renumbering is not cosmetic -- it's required for correctness: _should_print()/
    evaluate_generation() derive the NEXT evaluation_idx from
    len(_REWARD_EVOLUTION["generation"]) + 1, not from an independent counter. Popping
    an entry off the front shrinks that length by 1, so without also shifting every
    remaining label down by 1, the very next real generation would recompute the exact
    evaluation_idx the just-discarded entry already used (and already printed) --
    silently colliding two DIFFERENT generations onto the same label instead of just
    leaving a gap. Shifting labels down keeps "list length == most recent generation
    label" true, the same invariant that held before anything was ever discarded.

    For evotorch's Cosyne specifically (see runners/run_evolution.py): its searcher's
    very first .step() call evaluates the raw initial population once, BEFORE doing its
    first real generation of selection/variation -- an extra, structurally different
    (differently-sized, differently-composed) evaluation that both this function's
    caller and evaluate_generation()/_record_history() above have no way to tell apart
    from a real generation ahead of time, since it reaches this module through the exact
    same fitness_function() entrypoint. Every OTHER algorithm here (PGPE; any future
    single-shared-search-distribution or population-based method) calls evaluate() ==
    exactly one generation for every real searcher.step(), so this is never needed for
    them -- run_evolution.py calls this only from Cosyne's es_method branch, only once,
    right after that first step()."""
    global _REWARD_EVOLUTION, _TRACKED_GENERATIONS, _TRACKED_RECORDS
    global _CUE_IMPORTANCE_HISTORY, _TRANSFER_METRICS_HISTORY

    discarded_generation = _REWARD_EVOLUTION["generation"][0]
    for key in _REWARD_EVOLUTION:
        _REWARD_EVOLUTION[key].pop(0)
    for idx in range(len(_REWARD_EVOLUTION["generation"])):
        _REWARD_EVOLUTION["generation"][idx] -= 1

    if _TRACKED_GENERATIONS and _TRACKED_GENERATIONS[0] == discarded_generation:
        _TRACKED_GENERATIONS.pop(0)
        _TRACKED_RECORDS.pop(0)
    for idx in range(len(_TRACKED_GENERATIONS)):
        _TRACKED_GENERATIONS[idx] -= 1
        _TRACKED_RECORDS[idx]["generation"] -= 1

    if _CUE_IMPORTANCE_HISTORY["generation"] and _CUE_IMPORTANCE_HISTORY["generation"][0] == discarded_generation:
        for key in _CUE_IMPORTANCE_HISTORY:
            _CUE_IMPORTANCE_HISTORY[key].pop(0)
    for idx in range(len(_CUE_IMPORTANCE_HISTORY["generation"])):
        _CUE_IMPORTANCE_HISTORY["generation"][idx] -= 1

    if _TRANSFER_METRICS_HISTORY["generation"] and _TRANSFER_METRICS_HISTORY["generation"][0] == discarded_generation:
        for key in _TRANSFER_METRICS_HISTORY:
            _TRANSFER_METRICS_HISTORY[key].pop(0)
    for idx in range(len(_TRANSFER_METRICS_HISTORY["generation"])):
        _TRANSFER_METRICS_HISTORY["generation"][idx] -= 1


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
    if evaluation_idx <= _FORCE_TRACK_FIRST_N:
        return True
    if evaluation_idx == _TOTAL_GENERATIONS:
        return True
    return evaluation_idx % _PRINT_INTERVAL == 0


def _task_event_tensors(tracking, paradigm_phases):
    """{phase_type: (decisions, correct_arm)}, each tensor [pop, task_runs] --
    tracking's tensors are [pop, total_runs], concatenated across every non-replay phase
    IN ORDER (see _concat_tracking_segments); this slices that back apart by phase_type
    and concatenates every occurrence of the SAME task together (a paradigm can repeat
    trainA/trainB more than once, e.g. "trainA,20,trainB,50,trainA,20,trainB,7" -- all
    trainA occurrences are judged as one task, not separately)."""
    run_start = 0
    ranges_by_task = {}
    for phase_type, value in paradigm_phases:
        if phase_type != PHASE_REPLAY:
            ranges_by_task.setdefault(phase_type, []).append((run_start, run_start + value))
        run_start += value

    tensors_by_task = {}
    for phase_type, ranges in ranges_by_task.items():
        decisions = torch.cat([tracking["decisions_by_run"][:, start:end] for start, end in ranges], dim=1)
        correct_arm = torch.cat([tracking["correct_arm_by_run"][:, start:end] for start, end in ranges], dim=1)
        tensors_by_task[phase_type] = (decisions, correct_arm)
    return tensors_by_task


def _check_event_count_termination(evaluation_idx, tracking, paradigm_phases):
    """Per task (trainA/trainB judged separately -- see _task_event_tensors), across
    the WHOLE tracked population (every network, every run of that task): if ANY of
    the following holds, the search has collapsed onto a degenerate policy that isn't
    going to develop further, and _TERMINATE_EARLY is set.
      - zero left turns, or zero right turns
      - zero CORRECT left turns, or zero CORRECT right turns (only checked when that
        direction was turned at all -- otherwise it's already covered by the point above)
    Population-wide and per-task on purpose: looking only at the best individual (the
    old plateau criterion) or only at counts combined across both tasks can each hide a
    genuinely-collapsed task behind a healthy-looking aggregate -- e.g. a population
    that always turns left (no penalties, but only half the reward available) can
    outcompete one that's actually learning to read the cue but still crashes often.

    Deliberately does NOT terminate on "every run crashed" (e.g. correctly turning but
    then crashing right after every time) -- unlike the two turn-direction conditions
    above, that's not one atomic behavior either happening or not: reaching mazeend
    needs FIVE separate dead-zone ticks plus the correct turn to all hold in the same
    run, and post-turn survival is a genuinely harder, later-arriving sub-problem than
    pre-turn survival (the CTRNN state is never reset, so it's still recovering from
    the excursion the turn itself just required) -- not evidence the search has
    collapsed, just that this particular piece hasn't fallen into place yet.

    Skips generation 1 entirely: initial weights are drawn small/near-zero (see
    genome_codec.py's sample_initial_weights), so a fresh population's turn-tick output
    is almost always inside the dead zone and crashes -- near-100% crashes at generation
    1 is the normal starting point of every run, not a collapsed search, and firing here
    would end runs before evolution gets any chance to act at all.

    _EARLY_TERMINATION_ENABLED (from run_evolution.py's CLI, via configure_printing) gates
    whether a met criterion actually halts the run: when False, the criterion below is
    still detected and printed every time it's met -- so a stuck-looking run is still
    visible in the terminal log -- but _TERMINATE_EARLY is never set, so
    run_evolution.py's main loop always runs the full configured generation count
    regardless. For deliberately forcing a run past what would normally look like a
    collapsed population, e.g. to see whether it recovers given more generations."""
    global _TERMINATE_EARLY
    if evaluation_idx == 1:
        return
    for phase_type, (decisions, correct_arm) in _task_event_tensors(tracking, paradigm_phases).items():
        left_turns = decisions == 0
        right_turns = decisions == 1

        reasons = []
        if not bool(left_turns.any().item()):
            reasons.append("zero left turns")
        if not bool(right_turns.any().item()):
            reasons.append("zero right turns")
        if bool(left_turns.any().item()) and not bool((left_turns & correct_arm).any().item()):
            reasons.append("zero CORRECT left turns")
        if bool(right_turns.any().item()) and not bool((right_turns & correct_arm).any().item()):
            reasons.append("zero CORRECT right turns")

        if not reasons:
            continue

        if _EARLY_TERMINATION_ENABLED:
            _TERMINATE_EARLY = True
            print(f"EARLY TERMINATION: task {phase_type} at generation {evaluation_idx} -- "
                  f"{', '.join(reasons)} -- across the whole tracked population. Collapsed onto "
                  "a degenerate policy, not going to develop further. Stopping here as if this "
                  "were the final generation.")
        else:
            print(f"EARLY TERMINATION CRITERION MET (disabled, continuing): task {phase_type} at "
                  f"generation {evaluation_idx} -- {', '.join(reasons)} -- across the whole tracked "
                  "population. Would normally stop here, but early termination is disabled -- "
                  "running the full configured generation count regardless.")


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
    paradigm_phases,
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
    _check_event_count_termination(evaluation_idx, tracking, paradigm_phases)
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
    variant_idx_cpu,
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
            # which paradigm variant each individual (row) drew this generation -- one entry
            # per individual, not per run (see evaluate_generation's variant_idx); run_evolution.py's
            # input-distributions plot only uses this when there's more than one variant to begin with
            "paradigm_variant_by_individual": variant_idx_cpu.clone(),
        }
    )


def _record_cue_importance(evaluation_idx, context_importance, sensory_importance, reward_importance):
    """Store one tracked generation's context/sensory/reward cue importance distribution."""
    context_cpu = context_importance.detach().cpu()
    sensory_cpu = sensory_importance.detach().cpu()
    reward_cpu = reward_importance.detach().cpu()
    _CUE_IMPORTANCE_HISTORY["generation"].append(evaluation_idx)
    _CUE_IMPORTANCE_HISTORY["context_importance_mean"].append(float(context_cpu.mean().item()))
    _CUE_IMPORTANCE_HISTORY["context_importance_min"].append(float(context_cpu.min().item()))
    _CUE_IMPORTANCE_HISTORY["context_importance_max"].append(float(context_cpu.max().item()))
    _CUE_IMPORTANCE_HISTORY["sensory_importance_mean"].append(float(sensory_cpu.mean().item()))
    _CUE_IMPORTANCE_HISTORY["sensory_importance_min"].append(float(sensory_cpu.min().item()))
    _CUE_IMPORTANCE_HISTORY["sensory_importance_max"].append(float(sensory_cpu.max().item()))
    _CUE_IMPORTANCE_HISTORY["reward_importance_mean"].append(float(reward_cpu.mean().item()))
    _CUE_IMPORTANCE_HISTORY["reward_importance_min"].append(float(reward_cpu.min().item()))
    _CUE_IMPORTANCE_HISTORY["reward_importance_max"].append(float(reward_cpu.max().item()))


def _record_transfer_metrics(evaluation_idx, metrics):
    """Store one tracked generation's FWT/BWT distribution (population-wide mean/min/max)
    plus each underlying R-component's population mean, for CSV traceability -- see
    _measure_transfer_metrics for what each component means. The component key set's
    size/names depend on this run's task count T (fixed for the whole run, since the
    paradigm doesn't change generation to generation) -- new keys are added to
    _TRANSFER_METRICS_HISTORY the first time they're seen (generation 1, always tracked),
    so every tracked generation ends up with the same key set with no gaps."""
    fwt_cpu = metrics["fwt"].detach().cpu()
    bwt_cpu = metrics["bwt"].detach().cpu()
    _TRANSFER_METRICS_HISTORY["generation"].append(evaluation_idx)
    _TRANSFER_METRICS_HISTORY["fwt_mean"].append(float(fwt_cpu.mean().item()))
    _TRANSFER_METRICS_HISTORY["fwt_min"].append(float(fwt_cpu.min().item()))
    _TRANSFER_METRICS_HISTORY["fwt_max"].append(float(fwt_cpu.max().item()))
    _TRANSFER_METRICS_HISTORY["bwt_mean"].append(float(bwt_cpu.mean().item()))
    _TRANSFER_METRICS_HISTORY["bwt_min"].append(float(bwt_cpu.min().item()))
    _TRANSFER_METRICS_HISTORY["bwt_max"].append(float(bwt_cpu.max().item()))
    for label, tensor in metrics["components"].items():
        key = f"{label}_mean"
        _TRANSFER_METRICS_HISTORY.setdefault(key, []).append(float(tensor.detach().cpu().mean().item()))


# ==== 6) PARADIGM EXECUTION =====================================================
def _concat_tracking_segments(segments):
    """Concatenate per-training-phase tracking dicts along the run axis (dim=1),
    so multiple trainA/trainB phases appear as one chronological run sequence."""
    return {key: torch.cat([segment[key] for segment in segments], dim=1) for key in segments[0]}


def _run_paradigm(genome, paradigm_phases, device, noise_generator, reward_generator, collect_tracking,
                   context_cues_on, sensory_cues_on, reward_cues_on, state, W):
    """Runs every (phase_type, value) in paradigm_phases in order, starting from the
    given (state, W) and chaining CTRNN state/weights across phases. Returns the
    final (state, W), the summed training and replay reward, per-training-task-type
    reward (training_reward_by_task -- occurrences of the same task type, e.g. two
    separate trainA phases, are summed together, same convention as fitness.py's
    _task_event_tensors), and checkpoint_after_task -- a {phase_type: (state, W)} dict
    capturing the checkpoint right when EACH distinct training phase type finishes (first
    occurrence only), used by _measure_transfer_metrics for its zero-shot probes; empty
    if paradigm_phases has no training phase at all. Never mutates the state/W tensors
    passed in -- simulate_training_phase/simulate_replay_phase both clone-or-recompute
    rather than mutate in place -- so the same (state, W) can safely be reused as a
    starting checkpoint across multiple independent calls."""
    training_reward = torch.zeros(state.shape[0], device=device)
    replay_reward = torch.zeros(state.shape[0], device=device)
    tracking_segments = []
    training_reward_by_task = {}
    checkpoint_after_task = {}

    for phase_type, value in paradigm_phases:
        if phase_type in PHASE_CONTEXT:
            result = simulate_training_phase(
                state, W, genome["M"], genome["A"], genome["B"], genome["C"], genome["D"],
                genome["beta"], genome["eta"], PHASE_CONTEXT[phase_type], value,
                context_cues_on, sensory_cues_on, reward_cues_on,
                noise_generator, reward_generator, device,
                collect_tracking=collect_tracking,
            )
            if collect_tracking:
                state, W, phase_reward, phase_tracking = result
                tracking_segments.append(phase_tracking)
            else:
                state, W, phase_reward = result
            training_reward = training_reward + phase_reward
            training_reward_by_task[phase_type] = training_reward_by_task.get(
                phase_type, torch.zeros_like(phase_reward)
            ) + phase_reward
            if phase_type not in checkpoint_after_task:
                checkpoint_after_task[phase_type] = (state.clone(), W.clone())
        elif phase_type == PHASE_REPLAY:
            state, W, replay_trace = simulate_replay_phase(
                state, W, genome["M"], genome["A"], genome["B"], genome["C"], genome["D"],
                genome["beta"], genome["eta"], value, noise_generator, device,
            )
            replay_reward = replay_reward + assign_replay_reward(replay_trace, REPLAY_REWARD_METHOD)
        else:
            raise ValueError(f"Unknown phase type '{phase_type}' in paradigm.")

    tracking = _concat_tracking_segments(tracking_segments) if collect_tracking else None
    return state, W, training_reward, replay_reward, tracking, training_reward_by_task, checkpoint_after_task


def _run_paradigm_per_individual(genome, paradigm_variants, variant_idx, device, noise_generator,
                                  reward_generator, collect_tracking, context_cues_on, sensory_cues_on,
                                  reward_cues_on, state, W):
    """Like _run_paradigm, but each individual (row) runs its OWN paradigm variant --
    variant_idx[row] (values 0..len(paradigm_variants)-1) selects which entry of
    paradigm_variants that individual drew (see evaluate_generation; variant_idx is
    all-zeros -- one variant for the whole population -- whenever this run's config only
    has one paradigm). Every variant is required to have the identical (phase, VALUE)
    shape (see paradigm.parse_paradigm_variants), so every output tensor below has the
    same fixed shape no matter how the population's variant draw came out.

    Splits the population into one GPU sub-batch per DISTINCT variant actually drawn
    this call (boolean mask, no Python-level per-individual loop), runs _run_paradigm on
    each sub-batch with that variant's own phase list, and scatters every result back
    into full-population-sized tensors at the original row positions. Total compute is
    the same order as a single fused call over the whole population would have been (sum
    over variants of sub_pop * that variant's tick count) -- this never leaves the GPU,
    it just replaces one fused call with len(paradigm_variants) sequential vectorized
    calls (typically a handful)."""
    pop = state.shape[0]
    state_final = torch.empty_like(state)
    W_final = torch.empty_like(W)
    training_reward = torch.empty(pop, device=device)
    replay_reward = torch.empty(pop, device=device)
    training_reward_by_task = {}
    checkpoint_after_task = {}
    tracking = None

    for variant, paradigm_phases in enumerate(paradigm_variants):
        rows = torch.where(variant_idx == variant)[0]
        if rows.numel() == 0:
            continue

        sub_genome = {name: tensor[rows] for name, tensor in genome.items()}
        (
            sub_state_final, sub_W_final, sub_training_reward, sub_replay_reward,
            sub_tracking, sub_training_reward_by_task, sub_checkpoint_after_task,
        ) = _run_paradigm(
            sub_genome, paradigm_phases, device, noise_generator, reward_generator, collect_tracking,
            context_cues_on, sensory_cues_on, reward_cues_on, state[rows], W[rows],
        )

        state_final[rows] = sub_state_final
        W_final[rows] = sub_W_final
        training_reward[rows] = sub_training_reward
        replay_reward[rows] = sub_replay_reward

        for phase_type, reward_by_task in sub_training_reward_by_task.items():
            training_reward_by_task.setdefault(phase_type, torch.zeros(pop, device=device))[rows] = reward_by_task

        for phase_type, (chk_state, chk_W) in sub_checkpoint_after_task.items():
            full_state, full_W = checkpoint_after_task.setdefault(
                phase_type, (torch.zeros_like(state), torch.zeros_like(W))
            )
            full_state[rows] = chk_state
            full_W[rows] = chk_W

        if collect_tracking:
            if tracking is None:
                tracking = {
                    key: torch.zeros((pop,) + value.shape[1:], dtype=value.dtype, device=value.device)
                    for key, value in sub_tracking.items()
                }
            for key, value in sub_tracking.items():
                tracking[key][rows] = value

    return state_final, W_final, training_reward, replay_reward, tracking, training_reward_by_task, checkpoint_after_task


def _measure_cue_importance(genome, paradigm_variants, variant_idx, device, test_generator,
                             context_cues_on, sensory_cues_on, state_checkpoint, W_checkpoint):
    """Ablation importance: reward lost when one cue channel is clipped to zero,
    measured by continuing the SAME paradigm shape once more from the real
    post-evaluation (state_checkpoint, W_checkpoint) -- i.e. probing what the
    actually-trained network does, not a network retrained from scratch.

    Uses test_generator exclusively (never the evolutionary noise/reward
    generators), so this diagnostic draws no randomness from and has zero effect
    on the main evolutionary RNG stream. All four probes (baseline + three
    ablations) are replayed from the identical test_generator state, so they see
    the same maze draws/noise and differ only in which cue is ablated. Every
    probe's resulting state/W is discarded once its reward is read out -- nothing
    from testing carries out into the real evaluation, only the checkpoint carries in.

    The reward-signal probe always ablates from a baseline with context/sensory ON --
    the online reward input (constants.INPUT_REWARD) is never itself gated by
    context_cues_on/sensory_cues_on, so this is the one ablation not tied to those flags.
    """
    checkpoint_rng_state = test_generator.get_state()

    def _probe(probe_context_cues_on, probe_sensory_cues_on, probe_reward_cues_on):
        test_generator.set_state(checkpoint_rng_state)
        _, _, probe_training_reward, probe_replay_reward, _, _, _ = _run_paradigm_per_individual(
            genome, paradigm_variants, variant_idx, device, test_generator, test_generator, False,
            probe_context_cues_on, probe_sensory_cues_on, probe_reward_cues_on,
            state_checkpoint, W_checkpoint,
        )
        return probe_training_reward + probe_replay_reward

    baseline_reward = _probe(context_cues_on, sensory_cues_on, True)
    context_ablated_reward = _probe(False, sensory_cues_on, True)
    sensory_ablated_reward = _probe(context_cues_on, False, True)
    reward_ablated_reward = _probe(context_cues_on, sensory_cues_on, False)

    context_importance = baseline_reward - context_ablated_reward
    sensory_importance = baseline_reward - sensory_ablated_reward
    reward_importance = baseline_reward - reward_ablated_reward
    return context_importance, sensory_importance, reward_importance


def paradigm_has_multiple_training_tasks(paradigm_phases):
    """True when paradigm_phases has at least two DISTINCT training task types -- the
    minimum FWT/BWT (_measure_transfer_metrics) needs (see _training_task_order). A
    single-training-task paradigm (whether that task appears once or is split across
    several occurrences, e.g. by a replay bout in between) has no transfer to measure --
    callers use this to skip transfer-metric tracking entirely for such a paradigm
    instead of hitting _training_task_order's loud ValueError, which stays reserved for
    the genuinely ambiguous case (two-or-more distinct tasks where one repeats)."""
    task_types = [phase_type for phase_type, _ in paradigm_phases if phase_type in PHASE_CONTEXT]
    return len(set(task_types)) >= 2


def _training_task_order(paradigm_phases):
    """Ordered list of (task_type, num_runs), one entry per DISTINCT training task in
    paradigm_phases, in chronological order of occurrence -- e.g. [("trainA", 50),
    ("trainB", 50)] or, once a third task type exists (e.g. a double-T-maze), [("trainA",
    ...), ("trainB", ...), ("trainC", ...)]. FWT/BWT (_measure_transfer_metrics) are
    defined for any task count T >= 2 (see Lopez-Paz & Ranzato 2017's R in R^{T x T});
    T is read from the paradigm every call, never hardcoded, so a third training task
    added to paradigm.py's VALID_PHASE_TYPES/this module's PHASE_CONTEXT needs no other
    change here or downstream (recording/h5/plot/CSV all adapt automatically).

    Assumes each distinct task appears exactly once (repeats aren't supported by this
    metric yet) and that there are at least two distinct tasks -- raises loudly rather
    than silently computing something meaningless otherwise (a single-task paradigm has
    no transfer to measure; a repeated task's checkpoints would be ambiguous)."""
    training_phases = [(phase_type, value) for phase_type, value in paradigm_phases if phase_type in PHASE_CONTEXT]
    task_types = [phase_type for phase_type, _ in training_phases]
    if len(task_types) < 2 or len(set(task_types)) != len(task_types):
        raise ValueError(
            "FWT/BWT transfer metrics (see project-plan point 6.2, Lopez-Paz & Ranzato 2017) "
            "need at least two DISTINCT training tasks, each appearing exactly once in the "
            f"paradigm; got training phases {training_phases} from paradigm {paradigm_phases}."
        )
    return training_phases


def _measure_transfer_metrics(genome, device, test_generator, context_cues_on, sensory_cues_on,
                               state0, W_init, checkpoint_after_task, state_final, W_final,
                               task_order, training_reward_by_task):
    """Forward/backward knowledge transfer, exactly as defined in Lopez-Paz, D., & Ranzato,
    M. (2017), "Gradient Episodic Memory for Continual Learning" (NeurIPS 2017), Section 2,
    Equations 3-4, generalized to however many distinct training tasks this paradigm
    actually has (T = len(task_order) >= 2 -- see _training_task_order). Task order is
    read from the paradigm itself, NOT hardcoded to any particular sequence, so an
    A-then-B, B-then-A, or (future) A-then-B-then-C paradigm all report correctly.

    Paper's matrix: R_i,j = test performance on task j after observing the last sample of
    task i; b-bar_j = performance on task j at random initialization. Our "performance" is
    mean maze-task reward per run (see project-plan 6.1: performance is reward collected
    only, never the L1-/replay-adjusted evolutionary fitness). This project's paradigm
    trains each distinct task exactly once, in the fixed chronological order task_order,
    so "checkpoint after task i" (i = 1..T, in that chronological order) is unambiguous:

        R_i,i   = task i performance, measured right after task i is trained (the real
                  run's own segment reward for that task -- training_reward_by_task, no
                  extra probe needed)
        R_T,i   (i = 1..T-1) = task i performance measured after ALL T tasks are trained
                  -- a final-test probe from state_final/W_final
        R_i-1,i (i = 2..T)   = task i performance measured right after task (i-1) is
                  trained, BEFORE task i has been trained at all -- a zero-shot probe
                  from checkpoint_after_task[task (i-1)]
        b_i     (i = 1..T)   = task i performance at random initialization (state0/
                  W_init) -- baseline probe

        BWT = (1/(T-1)) * sum_{i=1}^{T-1} (R_T,i - R_i,i)    (paper Eq. 3)
        FWT = (1/(T-1)) * sum_{i=2}^{T}   (R_i-1,i - b_i)    (paper Eq. 4)

    Every probe runs through test_generator only (never the evolutionary noise/reward
    generators), exactly like _measure_cue_importance -- they read the network's state
    but leave no trace on the real evaluation. collect_tracking=False: only the scalar
    reward is needed, not per-run decision tracking.

    Returns {"bwt": tensor, "fwt": tensor, "components": {label: tensor}}. "components"
    holds every underlying R_i,i / R_T,i / R_i-1,i / b_i probe, keyed by a name built
    from this run's actual task labels (e.g. "R_A_own", "baseline_B", "R_A_then_B_zero_
    shot") -- purely for CSV/h5 traceability, and naturally sized to however many terms
    T-1 and T actually produce (2 tasks -> 4 components, 3 tasks -> 7, ...).
    """
    def _probe(phase_type, num_runs, state, W):
        _, _, probe_training_reward, _, _, _, _ = _run_paradigm(
            genome, [(phase_type, num_runs)], device, test_generator, test_generator, False,
            context_cues_on, sensory_cues_on, True, state, W,
        )
        return probe_training_reward / num_runs

    num_tasks = len(task_order)
    labels = [PHASE_CONTEXT[task_type] for task_type, _ in task_order]
    components = {}

    r_diag = {}
    baseline = {}
    for (task_type, num_runs), label in zip(task_order, labels):
        r_diag[task_type] = training_reward_by_task[task_type] / num_runs
        components[f"R_{label}_own"] = r_diag[task_type]
        baseline[task_type] = _probe(task_type, num_runs, state0, W_init)
        components[f"baseline_{label}"] = baseline[task_type]

    bwt_terms = []
    for (task_type, num_runs), label in zip(task_order[:-1], labels[:-1]):
        r_final_test = _probe(task_type, num_runs, state_final, W_final)
        components[f"R_{label}_final_test"] = r_final_test
        bwt_terms.append(r_final_test - r_diag[task_type])
    bwt = sum(bwt_terms) / (num_tasks - 1)

    fwt_terms = []
    for idx in range(1, num_tasks):
        prev_task_type, prev_label = task_order[idx - 1][0], labels[idx - 1]
        task_type, num_runs, label = task_order[idx][0], task_order[idx][1], labels[idx]
        r_zero_shot = _probe(task_type, num_runs, *checkpoint_after_task[prev_task_type])
        components[f"R_{prev_label}_then_{label}_zero_shot"] = r_zero_shot
        fwt_terms.append(r_zero_shot - baseline[task_type])
    fwt = sum(fwt_terms) / (num_tasks - 1)

    return {"bwt": bwt, "fwt": fwt, "components": components}


# ==== 7) FITNESS EVALUATION =====================================================
def evaluate_generation(genome_flat, device, noise_generator, reward_generator, test_generator,
                         weight_init_generator, l1_lambda, context_cues_on, sensory_cues_on,
                         evo_plasticity_on, paradigm_variants, paradigm_generator):
    evaluation_idx = len(_REWARD_EVOLUTION["generation"]) + 1
    should_print = _should_print(evaluation_idx)

    genome_flat = genome_flat.clone()
    pop = genome_flat.shape[0]
    genome = unflatten_genome(genome_flat, pop)
    if not evo_plasticity_on:
        genome["eta"] = torch.zeros_like(genome["eta"])

    # fresh initial weights every lifetime -- NOT read from the genome, see genome_codec.py
    W_init = sample_initial_weights(pop, device, weight_init_generator)

    # Which paradigm variant each individual gets, redrawn fresh every generation --
    # no coordination within OR across generations is needed (see paradigm.parse_paradigm_variants),
    # so this is simply "at birth" for every individual regardless of es_method: a freshly
    # sampled PGPE individual is genuinely born this generation, and a persisting Cosyne
    # individual doesn't need its paradigm to stay the same across its lifetime either.
    # All-zeros (skip the draw, no dedicated RNG stream needed) when there's only one
    # variant -- the overwhelmingly common case -- so single-paradigm runs/configs are
    # completely unaffected, including their RNG consumption.
    if len(paradigm_variants) == 1:
        variant_idx = torch.zeros(pop, dtype=torch.long, device=device)
    else:
        variant_idx = torch.randint(0, len(paradigm_variants), (pop,), generator=paradigm_generator, device=device)
    representative_paradigm_phases = paradigm_variants[0]  # structural stand-in for printing/task-order
    # purposes only -- every variant has the identical (phase, VALUE) shape by construction,
    # so positions/run-counts read off this one are correct for the whole population even
    # though its phase TYPE labels may not describe every individual's actual variant

    frob_start = torch.linalg.matrix_norm(W_init, ord="fro", dim=(1, 2))
    state0 = torch.zeros(pop, constants.N, device=device)
    state_final, W_final, training_reward, replay_reward, tracking, training_reward_by_task, checkpoint_after_task = _run_paradigm_per_individual(
        genome, paradigm_variants, variant_idx, device, noise_generator, reward_generator, should_print,
        context_cues_on, sensory_cues_on, True, state0, W_init,  # reward cue is always on
        # during real evolution -- ablating it is only ever a diagnostic probe, see
        # _measure_cue_importance, not an evolutionary condition (no evo_reward_cues_on exists)
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
            representative_paradigm_phases,
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
            variant_idx.detach().cpu(),
        )
        context_importance, sensory_importance, reward_importance = _measure_cue_importance(
            genome, paradigm_variants, variant_idx, device, test_generator,
            context_cues_on, sensory_cues_on, state_final, W_final,
        )
        _record_cue_importance(evaluation_idx, context_importance, sensory_importance, reward_importance)

        # Single-training-task paradigms have no transfer to measure -- skip rather
        # than hit _training_task_order's loud ValueError, see paradigm_has_multiple_training_tasks.
        # (Safe to read off representative_paradigm_phases alone: whenever transfer metrics
        # are actually wanted, run_evolution.py requires every variant to share the same
        # task-type sequence, not just the same shape -- see its HAS_TRANSFER_METRICS check.)
        if paradigm_has_multiple_training_tasks(representative_paradigm_phases):
            task_order = _training_task_order(representative_paradigm_phases)
            transfer_metrics = _measure_transfer_metrics(
                genome, device, test_generator, context_cues_on, sensory_cues_on,
                state0, W_init, checkpoint_after_task, state_final, W_final,
                task_order, training_reward_by_task,
            )
            _record_transfer_metrics(evaluation_idx, transfer_metrics)
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
            variant_idx.detach().cpu(),
        )

    return regularized_fitness


def fitness_function(genome_flat, device, noise_generator, reward_generator, test_generator,
                      weight_init_generator, l1_lambda, context_cues_on, sensory_cues_on,
                      evo_plasticity_on, paradigm_variants, paradigm_generator):
    """Vectorized EvoTorch objective entrypoint."""
    return evaluate_generation(genome_flat, device, noise_generator, reward_generator, test_generator,
                                weight_init_generator, l1_lambda, context_cues_on, sensory_cues_on,
                                evo_plasticity_on, paradigm_variants, paradigm_generator)