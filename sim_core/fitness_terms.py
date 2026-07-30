"""Reusable scalar fitness terms."""

import torch


def compute_l1_penalty(genome_flat, l1_lambda):
    """Return (complexity, penalty) vectors with one value per individual."""
    if l1_lambda < 0:
        raise ValueError("l1_lambda must be non-negative.")
    complexity = genome_flat.abs().mean(dim=1)
    penalty = l1_lambda * complexity
    return complexity, penalty
