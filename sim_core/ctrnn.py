"""Batched CTRNN activation and neuromodulated plasticity update, 1:1 with the skeleton equations."""

import torch
from sim_core import constants


def activation_step(state, W, beta, noise_std, generator):
    """One CTRNN tick for ALL neurons. Caller is responsible for re-clamping
    input-neuron entries afterward, since this applies the ODE update to every neuron."""
    net_input = torch.einsum("bij,bj->bi", W, state) + beta
    noise = torch.randn(net_input.shape, generator=generator, device=net_input.device) * noise_std
    new_state = state + (constants.DT / constants.TAU) * (-state + torch.tanh(net_input + noise))
    return new_state


def plasticity_step(state, W, M, A, B, C, D, eta):
    """state here is the OLD (pre-update) state -- used as both pre- and post-synaptic activity."""
    pre = state
    post = state

    mod_signal = torch.einsum("bkij,bk->bij", M, state)
    mod_term = torch.tanh(mod_signal / 2)

    term_AB = torch.einsum("bij,bj,bi->bij", A, pre, post)
    term_B = torch.einsum("bij,bj->bij", B, pre)
    term_C = torch.einsum("bij,bi->bij", C, post)
    hebbian = term_AB + term_B + term_C + D

    dW = eta.view(-1, 1, 1) * mod_term * hebbian
    return dW
