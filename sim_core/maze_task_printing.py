"""Printing/instrumented training-phase simulation.

This keeps the original dynamics intact while also collecting per-run decision traces.
"""

import torch
from sim_core.constants import (
    N_INPUT, OUTPUT_IDX, NOISE_STD, STRAIGHT_THRESH,
    BIG_REWARD, SMALL_REWARD, CRASH_PENALTY,
    NUM_RUNS_PER_TRAINING_PHASE, MAX_TRAINING_TICKS,
)
from sim_core.ctrnn import activation_step, plasticity_step


def _draw_reward_arm(pop, reward_generator, device):
    return torch.randint(0, 2, (pop,), generator=reward_generator, device=device)


def _sensory_from_arm(big_reward_arm, context_is_A):
    cue_arm = big_reward_arm if context_is_A else (1 - big_reward_arm)
    sensory_a = (cue_arm == 0).float()
    sensory_b = 1.0 - sensory_a
    return sensory_a, sensory_b


def simulate_training_phase_printing(state, W, M, A, B, C, D, beta, eta,
                                     context_is_A, noise_generator, reward_generator, device):
    """Run training phase and return tracking info in addition to state/W/reward."""
    pop = state.shape[0]
    state = state.clone()

    step_in_run = torch.ones(pop, dtype=torch.long, device=device)
    run_count = torch.zeros(pop, dtype=torch.long, device=device)
    chosen_arm = torch.zeros(pop, dtype=torch.long, device=device)
    current_turn_choice = torch.full((pop,), -1, dtype=torch.long, device=device)
    total_reward = torch.zeros(pop, device=device)

    decisions_by_run = torch.full((pop, NUM_RUNS_PER_TRAINING_PHASE), -1, dtype=torch.long, device=device)
    crashed_by_run = torch.zeros((pop, NUM_RUNS_PER_TRAINING_PHASE), dtype=torch.bool, device=device)
    rewarded_by_run = torch.zeros((pop, NUM_RUNS_PER_TRAINING_PHASE), dtype=torch.bool, device=device)
    big_reward_by_run = torch.zeros((pop, NUM_RUNS_PER_TRAINING_PHASE), dtype=torch.bool, device=device)

    context_a = torch.full((pop,), 1.0 if context_is_A else 0.0, device=device)
    context_b = 1.0 - context_a

    big_reward_arm = _draw_reward_arm(pop, reward_generator, device)
    sensory_a, sensory_b = _sensory_from_arm(big_reward_arm, context_is_A)

    for _ in range(MAX_TRAINING_TICKS):
        active = run_count < NUM_RUNS_PER_TRAINING_PHASE
        if not torch.any(active):
            break

        is_home = (step_in_run == 1).float()
        is_turn_tick = step_in_run == 4
        is_end_tick = step_in_run == 7
        input_vec = torch.stack(
            [is_home, is_turn_tick.float(), is_end_tick.float(),
             context_a, context_b, sensory_a, sensory_b], dim=1,
        )

        state[:, :N_INPUT] = input_vec
        new_state = activation_step(state, W, beta, NOISE_STD, noise_generator)
        dW = plasticity_step(state, W, M, A, B, C, D, eta)
        W = W + dW
        new_state[:, :N_INPUT] = input_vec
        output = new_state[:, OUTPUT_IDX]

        straight_ok = output.abs() < STRAIGHT_THRESH
        turn_ok = output.abs() >= STRAIGHT_THRESH
        correct = torch.where(is_turn_tick, turn_ok, straight_ok)

        arm_choice = torch.where(output >= STRAIGHT_THRESH, torch.ones_like(chosen_arm), torch.zeros_like(chosen_arm))
        chosen_arm = torch.where(is_turn_tick, arm_choice, chosen_arm)
        current_turn_choice = torch.where(is_turn_tick, arm_choice, current_turn_choice)

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

        terminate = crash | got_reward
        finished = torch.where(terminate)[0]
        if finished.numel() > 0:
            finished_run_idx = run_count[finished]
            decisions_by_run[finished, finished_run_idx] = current_turn_choice[finished]
            crashed_by_run[finished, finished_run_idx] = crash[finished]
            rewarded_by_run[finished, finished_run_idx] = got_reward[finished]
            big_reward_by_run[finished, finished_run_idx] = (got_reward & chose_big_reward_arm)[finished]

        run_count += terminate.long()
        step_in_run = torch.where(terminate, torch.ones_like(step_in_run), step_in_run + 1)
        current_turn_choice = torch.where(terminate, torch.full_like(current_turn_choice, -1), current_turn_choice)

        new_arm = _draw_reward_arm(pop, reward_generator, device)
        big_reward_arm = torch.where(terminate, new_arm, big_reward_arm)
        new_sensory_a, new_sensory_b = _sensory_from_arm(big_reward_arm, context_is_A)
        sensory_a = torch.where(terminate, new_sensory_a, sensory_a)
        sensory_b = torch.where(terminate, new_sensory_b, sensory_b)

        state = new_state

    tracking = {
        "decisions_by_run": decisions_by_run,
        "crashed_by_run": crashed_by_run,
        "rewarded_by_run": rewarded_by_run,
        "big_reward_by_run": big_reward_by_run,
    }
    return state, W, total_reward, tracking
