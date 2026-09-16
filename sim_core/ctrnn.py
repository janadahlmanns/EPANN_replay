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


def plasticity_step(state, W, pre, post, M, A, B, C, D, eta):
    """state here is the OLD (pre-update) state, read only for the neuromodulatory
    signal (mod_signal) -- always instantaneous, never averaged (Dittrich's moving-average
    treatment applies to the Hebbian pre/post terms only, not the modulator). pre/post are
    supplied by the caller -- today that's just state again (MA_SPAN=1 degenerates to the
    original instantaneous-state behavior), but a caller can pass a moving average over
    the last few ticks instead (see maze_task.py/replay_task.py's tick loops) without this
    function needing to know the difference. eta is per-synapse ([pop, N, N], same shape
    as W), not a global scalar."""
    mod_signal = torch.einsum("bkij,bk->bij", M, state)
    mod_term = torch.tanh(mod_signal / 2)

    term_AB = torch.einsum("bij,bj,bi->bij", A, pre, post)
    term_B = torch.einsum("bij,bj->bij", B, pre)
    term_C = torch.einsum("bij,bi->bij", C, post)
    hebbian = term_AB + term_B + term_C + D

    dW = eta * mod_term * hebbian
    return dW
