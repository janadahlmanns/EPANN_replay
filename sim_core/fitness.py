"""Reshapes a flat genome batch, runs one training phase followed by one replay phase,
and returns total fitness. This is the EvoTorch objective function entry point."""

import torch
from sim_core.constants import N
from sim_core.genome_codec import unflatten_genome
from sim_core.fitness_terms import compute_l1_penalty
from sim_core.maze_task import simulate_training_phase
from sim_core.replay_task import simulate_replay_phase, assign_replay_reward

# ==== CONSTANTS ================================================================
REPLAY_REWARD_METHOD = "zero"      # placeholder; only method defined so far
TRAINING_CONTEXT_IS_A = True       # ASSUMPTION: single fixed context for this first pass
                                    # (no reversal / task-switching / counterbalancing yet)


# ==== FITNESS PIPELINE ==========================================================
def evaluate_generation(genome_flat, device, noise_generator, reward_generator, l1_lambda):
    """genome_flat: [pop, GENOME_LENGTH] -> total fitness [pop]."""
    genome_flat = genome_flat.clone()  # escape EvoTorch's ReadOnlyTensor before deriving anything from it
    pop = genome_flat.shape[0]
    genome = unflatten_genome(genome_flat, pop)

    state0 = torch.zeros(pop, N, device=device)

    state, W, training_reward = simulate_training_phase(
        state0, genome["W"], genome["M"], genome["A"], genome["B"], genome["C"], genome["D"],
        genome["beta"], genome["eta"], TRAINING_CONTEXT_IS_A,
        noise_generator, reward_generator, device,
    )

    _, _, replay_trace = simulate_replay_phase(
        state, W, genome["M"], genome["A"], genome["B"], genome["C"], genome["D"],
        genome["beta"], genome["eta"], noise_generator, device,
    )
    replay_reward = assign_replay_reward(replay_trace, REPLAY_REWARD_METHOD)
    _, l1_penalty = compute_l1_penalty(genome_flat, l1_lambda)

    return training_reward + replay_reward - l1_penalty


def fitness_function(genome_flat, device, noise_generator, reward_generator, l1_lambda):
    """Thin alias matching skeleton naming; bound via functools.partial in the runner
    since EvoTorch's vectorized objective_func takes a single tensor argument."""
    return evaluate_generation(genome_flat, device, noise_generator, reward_generator, l1_lambda=l1_lambda)