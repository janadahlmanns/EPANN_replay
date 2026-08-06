"""Flatten/unflatten genome tensors <-> a single 1D vector per individual, since EvoTorch's
PGPE operates on flat solution vectors and our fitness function needs structured tensors.

Per the Najarro & Risi (2020) meta-learning-through-plasticity framework, the genome
encodes only the plasticity rule (M, A, B, C, D, per-synapse eta) and tonic bias beta --
NEVER initial synaptic weights. Initial weights are sampled fresh every lifetime by
sample_initial_weights() below, outside the genome entirely."""

import torch
from sim_core.constants import N, WEIGHT_INIT_LOW, WEIGHT_INIT_HIGH

# ==== GENOME LAYOUT ==========================================================
GENOME_SPEC = [
    ("M", (N, N, N)),
    ("A", (N, N)),
    ("B", (N, N)),
    ("C", (N, N)),
    ("D", (N, N)),
    ("beta", (N,)),
    ("eta", (N, N)),
]
GENOME_SIZES = [int(torch.prod(torch.tensor(shape))) for _, shape in GENOME_SPEC]
GENOME_LENGTH = sum(GENOME_SIZES)


# ==== CODEC FUNCTIONS =========================================================
def flatten_genome(genome_dict, pop):
    """dict of [pop, *shape] tensors -> [pop, GENOME_LENGTH]."""
    parts = [genome_dict[name].reshape(pop, -1) for name, _ in GENOME_SPEC]
    return torch.cat(parts, dim=1)


def unflatten_genome(flat, pop):
    """[pop, GENOME_LENGTH] -> dict of [pop, *shape] tensors."""
    genome = {}
    offset = 0
    for (name, shape), size in zip(GENOME_SPEC, GENOME_SIZES):
        genome[name] = flat[:, offset:offset + size].reshape(pop, *shape)
        offset += size
    return genome


# ==== INITIAL-WEIGHT SAMPLING (NOT part of the genome) =========================
def sample_initial_weights(pop, device, generator):
    """Draw fresh initial synaptic weights W ~ U[WEIGHT_INIT_LOW, WEIGHT_INIT_HIGH] for
    every individual, every lifetime -- matches Najarro & Risi (2020)'s w ~ U[-0.1, 0.1]
    resampled at the start of every rollout. Uses its own dedicated RNG stream
    (generator) so this is reproducible independent of the noise/reward/test streams."""
    return torch.empty(pop, N, N, device=device).uniform_(
        WEIGHT_INIT_LOW, WEIGHT_INIT_HIGH, generator=generator
    )
