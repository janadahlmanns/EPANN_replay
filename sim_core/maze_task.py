"""Batched T-maze training phase (Soltoggio-style T-maze/reversal paradigm).

Each individual in the batch runs its own independent sequence of maze runs, all
stepped in lockstep tick-by-tick so the whole population stays vectorized on GPU.
A run is 7 ticks (home, straight, straight, turn, straight, straight, mazeend) if
never crashed; a wrong output at any tick ends the run immediately (crash penalty,
reset to start). The CTRNN state and weights W are NEVER reset between runs or
maze resets -- only the maze/task bookkeeping (position, reward arm, sensory cue)
resets. This is what makes the plasticity itself the thing being evolved.
"""

import torch
from sim_core.constants import (
    N_INPUT, OUTPUT_IDX, NOISE_STD, STRAIGHT_THRESH,
    BIG_REWARD, SMALL_REWARD, CRASH_PENALTY,
    NUM_RUNS_PER_TRAINING_PHASE, MAX_TRAINING_TICKS,
)
from sim_core.ctrnn import activation_step, plasticity_step


# ==== HELPERS =================================================================
def _draw_reward_arm(pop, reward_generator, device):
    return torch.randint(0, 2, (pop,), generator=reward_generator, device=device)


def _sensory_from_arm(big_reward_arm, context_is_A):
    """context_is_A: python bool, same context for the whole batch this call."""
    cue_arm = big_reward_arm if context_is_A else (1 - big_reward_arm)
    sensory_a = (cue_arm == 0).float()
    sensory_b = 1.0 - sensory_a
    return sensory_a, sensory_b


def _initialize_training_phase(state, context_is_A, reward_generator, device):
    pop = state.shape[0]
    state = state.clone()

    step_in_run = torch.ones(pop, dtype=torch.long, device=device)
    run_count = torch.zeros(pop, dtype=torch.long, device=device)
    chosen_arm = torch.zeros(pop, dtype=torch.long, device=device)
    total_reward = torch.zeros(pop, device=device)

    context_a = torch.full((pop,), 1.0 if context_is_A else 0.0, device=device)
    context_b = 1.0 - context_a

    big_reward_arm = _draw_reward_arm(pop, reward_generator, device)
    sensory_a, sensory_b = _sensory_from_arm(big_reward_arm, context_is_A)

    return (
        pop,
        state,
        step_in_run,
        run_count,
        chosen_arm,
        total_reward,
        context_a,
        context_b,
        big_reward_arm,
        sensory_a,
        sensory_b,
    )


# ==== MAIN SIMULATION ==========================================================
def simulate_training_phase(state, W, M, A, B, C, D, beta, eta,
                             context_is_A, noise_generator, reward_generator, device,
                             collect_tracking=False):
    """Runs NUM_RUNS_PER_TRAINING_PHASE maze runs (batched over population)."""
    (
        pop,
        state,
        step_in_run,
        run_count,
        chosen_arm,
        total_reward,
        context_a,
        context_b,
        big_reward_arm,
        sensory_a,
        sensory_b,
    ) = _initialize_training_phase(state, context_is_A, reward_generator, device)

    tracking = None
    current_turn_choice = None
    if collect_tracking:
        current_turn_choice = torch.full((pop,), -1, dtype=torch.long, device=device)
        tracking = {
            "decisions_by_run": torch.full((pop, NUM_RUNS_PER_TRAINING_PHASE), -1, dtype=torch.long, device=device),
            "crashed_by_run": torch.zeros((pop, NUM_RUNS_PER_TRAINING_PHASE), dtype=torch.bool, device=device),
            "rewarded_by_run": torch.zeros((pop, NUM_RUNS_PER_TRAINING_PHASE), dtype=torch.bool, device=device),
            "big_reward_by_run": torch.zeros((pop, NUM_RUNS_PER_TRAINING_PHASE), dtype=torch.bool, device=device),
            "sensory_cue_by_run": torch.zeros((pop, NUM_RUNS_PER_TRAINING_PHASE), dtype=torch.long, device=device),
        }
        tracking["sensory_cue_by_run"][:, 0] = (sensory_b > 0.5).long()

    for _ in range(MAX_TRAINING_TICKS):
        active = run_count < NUM_RUNS_PER_TRAINING_PHASE
        if not torch.any(active):
            break

        # --- build this tick's input vector ---
        is_home = (step_in_run == 1).float()
        is_turn_tick = step_in_run == 4
        is_end_tick = step_in_run == 7
        input_vec = torch.stack(
            [is_home, is_turn_tick.float(), is_end_tick.float(),
             context_a, context_b, sensory_a, sensory_b], dim=1,
        )

        # --- CTRNN tick: clamp inputs, advance state, apply plasticity ---
        state[:, :N_INPUT] = input_vec
        new_state = activation_step(state, W, beta, NOISE_STD, noise_generator)
        dW = plasticity_step(state, W, M, A, B, C, D, eta)
        W = W + dW
        W = W / W.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        new_state[:, :N_INPUT] = input_vec
        output = new_state[:, OUTPUT_IDX]

        # --- score this tick ---
        straight_ok = output.abs() < STRAIGHT_THRESH
        turn_ok = output.abs() >= STRAIGHT_THRESH
        correct = torch.where(is_turn_tick, turn_ok, straight_ok)

        arm_choice = torch.where(output >= STRAIGHT_THRESH, torch.ones_like(chosen_arm), torch.zeros_like(chosen_arm))
        chosen_arm = torch.where(is_turn_tick, arm_choice, chosen_arm)

        crash = active & (~correct)
        got_reward = active & is_end_tick & correct

        reward_delta = torch.zeros(pop, device=device)
        reward_delta = torch.where(crash, torch.full_like(reward_delta, CRASH_PENALTY), reward_delta)
        chose_big_reward_arm = chosen_arm == big_reward_arm
        arm_reward = torch.where(chose_big_reward_arm,
                                 torch.full_like(reward_delta, BIG_REWARD),
                                 torch.full_like(reward_delta, SMALL_REWARD))
        reward_delta = torch.where(got_reward, arm_reward, reward_delta)
        total_reward += reward_delta

        # --- terminate / reset finished runs ---
        terminate = crash | got_reward
        if collect_tracking:
            current_turn_choice = torch.where(is_turn_tick, arm_choice, current_turn_choice)
            finished = torch.where(terminate)[0]
            if finished.numel() > 0:
                finished_run_idx = run_count[finished]
                tracking["decisions_by_run"][finished, finished_run_idx] = current_turn_choice[finished]
                tracking["crashed_by_run"][finished, finished_run_idx] = crash[finished]
                tracking["rewarded_by_run"][finished, finished_run_idx] = got_reward[finished]
                tracking["big_reward_by_run"][finished, finished_run_idx] = (got_reward & chose_big_reward_arm)[finished]
            current_turn_choice = torch.where(terminate, torch.full_like(current_turn_choice, -1), current_turn_choice)

        run_count += terminate.long()
        step_in_run = torch.where(terminate, torch.ones_like(step_in_run), step_in_run + 1)

        new_arm = _draw_reward_arm(pop, reward_generator, device)
        big_reward_arm = torch.where(terminate, new_arm, big_reward_arm)
        new_sensory_a, new_sensory_b = _sensory_from_arm(big_reward_arm, context_is_A)
        sensory_a = torch.where(terminate, new_sensory_a, sensory_a)
        sensory_b = torch.where(terminate, new_sensory_b, sensory_b)

        if collect_tracking and finished.numel() > 0:
            next_run_idx = run_count[finished]
            valid = next_run_idx < NUM_RUNS_PER_TRAINING_PHASE
            if valid.any():
                vi = finished[valid]
                tracking["sensory_cue_by_run"][vi, next_run_idx[valid]] = (new_sensory_b[vi] > 0.5).long()

        state = new_state

    if collect_tracking:
        return state, W, total_reward, tracking
    return state, W, total_reward
