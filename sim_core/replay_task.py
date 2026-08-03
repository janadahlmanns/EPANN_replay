"""Batched replay phase: no maze, just quiescent (zero) input drive while plasticity
continues to run. Continues directly from the state/weights the training phase ended with."""

import torch
from sim_core.constants import N_INPUT, OUTPUT_IDX, NOISE_STD
from sim_core.ctrnn import activation_step, plasticity_step


def simulate_replay_phase(state, W, M, A, B, C, D, beta, eta, num_ticks, noise_generator, device):
    """Returns (state, W, output_trace) -- output_trace is [num_ticks, pop]."""
    pop = state.shape[0]
    state = state.clone()
    zero_input = torch.zeros(pop, N_INPUT, device=device)
    output_trace = torch.zeros(num_ticks, pop, device=device)

    for t in range(num_ticks):
        state[:, :N_INPUT] = zero_input
        new_state = activation_step(state, W, beta, NOISE_STD, noise_generator)
        dW = plasticity_step(state, W, M, A, B, C, D, eta)
        W = W + dW
        W = W / W.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
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
