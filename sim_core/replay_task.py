"""Batched replay phase: no maze, just quiescent (zero) input drive while plasticity
continues to run. Continues directly from the state/weights the training phase ended with."""

import collections

import torch
from sim_core import constants
from sim_core.constants import N_INPUT, OUTPUT_IDX
from sim_core.ctrnn import activation_step, plasticity_step


def simulate_replay_phase(state, W, M, A, B, C, D, beta, eta, num_ticks, noise_generator, device):
    """Returns (state, W, output_trace) -- output_trace is [num_ticks, pop].

    Hebbian update cadence/averaging (TAU_HEBB_MULT/MA_SPAN/WEIGHT_CLAMP, see
    constants.py) matches maze_task.py's simulate_training_phase exactly -- both the
    moving-average buffer and the tick counter start fresh here, same as that
    function's, and do not carry over across the trainA -> replay -> trainB phase
    boundary (see fitness.py's _run_paradigm, which only threads state/W between
    phases)."""
    pop = state.shape[0]
    state = state.clone()
    zero_input = torch.zeros(pop, N_INPUT, device=device)
    output_trace = torch.zeros(num_ticks, pop, device=device)

    pre_post_buffer = collections.deque(maxlen=constants.MA_SPAN)
    tick_idx = 0

    for t in range(num_ticks):
        state[:, :N_INPUT] = zero_input
        new_state = activation_step(state, W, beta, constants.NOISE_STD, noise_generator)

        pre_post_buffer.append(state)
        tick_idx += 1
        if tick_idx % constants.TAU_HEBB_MULT == 0:
            pre_post_avg = torch.stack(list(pre_post_buffer), dim=0).mean(dim=0)
            dW = plasticity_step(state, W, pre_post_avg, pre_post_avg, M, A, B, C, D, eta)
            W = W + dW
            clamp_scale = (W.abs().amax(dim=(1, 2), keepdim=True) / constants.WEIGHT_CLAMP).clamp(min=1.0)
            W = W / clamp_scale

        new_state[:, :N_INPUT] = zero_input
        output_trace[t] = new_state[:, OUTPUT_IDX]
        state = new_state

    return state, W, output_trace


def assign_replay_reward(replay_trace, method):
    """Placeholder for replay-reward hypotheses (spec item 2.4). 'zero' = no reward yet;
    future methods (e.g. rewarding replay of previously successful trajectories) go here."""
    pop = replay_trace.shape[1]
    if method == "zero":
        return torch.zeros(pop, device=replay_trace.device)
    raise ValueError(f"Unknown replay reward method: {method}")
