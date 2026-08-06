"""Flatten/unflatten genome tensors <-> a single 1D vector per individual, since EvoTorch's
PGPE operates on flat solution vectors and our fitness function needs structured tensors.

Per the Najarro & Risi (2020) meta-learning-through-plasticity framework, the genome
encodes only the plasticity rule (M, A, B, C, D, per-synapse eta) and tonic bias beta --
NEVER initial synaptic weights. Initial weights are sampled fresh every lifetime by
sample_initial_weights() below, outside the genome entirely.

Genome layout is computed from constants.N at CALL time, not import time -- N is a
per-run config value (see constants.configure_network()), and this module may be
imported before that runs (e.g. run_batch.py never calls configure_network() itself,
it only orchestrates subprocesses that do), so nothing here may depend on N at
module load time."""

import torch
from sim_core import constants

# ==== GENOME LAYOUT ==========================================================
def genome_spec():
    """Current genome layout as a list of (name, shape) tuples, sized by constants.N."""
    n = constants.N
    return [
        ("M", (n, n, n)),
        ("A", (n, n)),
        ("B", (n, n)),
        ("C", (n, n)),
        ("D", (n, n)),
        ("beta", (n,)),
        ("eta", (n, n)),
    ]


def genome_sizes():
    """Flat element count per genome_spec() entry, in the same order."""
    return [int(torch.prod(torch.tensor(shape))) for _, shape in genome_spec()]


def genome_length():
    """Total flat genome length for the current constants.N."""
    return sum(genome_sizes())


# ==== CODEC FUNCTIONS =========================================================
def flatten_genome(genome_dict, pop):
    """dict of [pop, *shape] tensors -> [pop, genome_length()]."""
    parts = [genome_dict[name].reshape(pop, -1) for name, _ in genome_spec()]
    return torch.cat(parts, dim=1)


def unflatten_genome(flat, pop):
    """[pop, genome_length()] -> dict of [pop, *shape] tensors."""
    genome = {}
    offset = 0
    for (name, shape), size in zip(genome_spec(), genome_sizes()):
        genome[name] = flat[:, offset:offset + size].reshape(pop, *shape)
        offset += size
    return genome


# ==== INITIAL-WEIGHT SAMPLING (NOT part of the genome) =========================
def sample_initial_weights(pop, device, generator):
    """Draw fresh initial synaptic weights W ~ U[WEIGHT_INIT_LOW, WEIGHT_INIT_HIGH] for
    every individual, every lifetime -- matches Najarro & Risi (2020)'s w ~ U[-0.1, 0.1]
    resampled at the start of every rollout. Uses its own dedicated RNG stream
    (generator) so this is reproducible independent of the noise/reward/test streams."""
    return torch.empty(pop, constants.N, constants.N, device=device).uniform_(
        constants.WEIGHT_INIT_LOW, constants.WEIGHT_INIT_HIGH, generator=generator
    )
