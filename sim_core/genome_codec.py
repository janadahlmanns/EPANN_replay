"""Flatten/unflatten genome tensors <-> a single 1D vector per individual, since EvoTorch's
PGPE operates on flat solution vectors and our fitness function needs structured tensors."""

import torch
from sim_core.constants import N

# ==== GENOME LAYOUT ==========================================================
GENOME_SPEC = [
    ("W", (N, N)),
    ("M", (N, N, N)),
    ("A", (N, N)),
    ("B", (N, N)),
    ("C", (N, N)),
    ("D", (N, N)),
    ("beta", (N,)),
    ("eta", (1,)),
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
