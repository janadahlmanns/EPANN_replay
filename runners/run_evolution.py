"""First raw end-to-end run: EvoTorch PGPE search over the flattened CTRNN genome,
using our saved test genome (individual 0) as the search center, for a handful of
generations, printing results. Not tuned/validated yet -- this is the "get it running"
pass; sparsity, hyperparameters, and task complexity (single fixed context, one
training+replay phase) are all deliberately minimal for now.
"""

# ==== 1. RNG DETERMINISM SETUP (must happen before any CUDA calls) ==========
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # also set this in your shell to be safe

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root, so sim_core is importable

import functools
import torch
from evotorch import Problem
from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger

from sim_core.genome_codec import GENOME_LENGTH, flatten_genome
from sim_core.fitness import fitness_function

# ==== 2. CONSTANTS / USER INPUTS =============================================
DEVICE = "cuda"
TEST_GENOME_PATH = "data/test_genome.pt"

MASTER_SEED = 0
NOISE_SEED = 1
REWARD_SEED = 2

NUM_GENERATIONS = 200
SEARCH_POPSIZE = 200           
CENTER_LEARNING_RATE = 0.1
STDEV_LEARNING_RATE = 0.1
STDEV_INIT = 0.5              # RESEARCH AND DECIDE 0.5 is just a guesss
MAX_SPEED = 0.15

torch.manual_seed(MASTER_SEED)
torch.cuda.manual_seed_all(MASTER_SEED)
torch.use_deterministic_algorithms(True)

noise_generator = torch.Generator(device=DEVICE)
noise_generator.manual_seed(NOISE_SEED)
reward_generator = torch.Generator(device=DEVICE)
reward_generator.manual_seed(REWARD_SEED)

# ==== 3. BUILD CENTER_INIT FROM THE SAVED TEST GENOME =======================
test_genome = torch.load(TEST_GENOME_PATH, map_location=DEVICE)
test_pop = test_genome["W"].shape[0]
center_init = flatten_genome(test_genome, test_pop)[0]   # individual 0 as the starting point

# ==== 4. BUILD EVOTORCH PROBLEM + PGPE SEARCHER ==============================
objective = functools.partial(
    fitness_function, device=DEVICE, noise_generator=noise_generator, reward_generator=reward_generator,
)

problem = Problem(
    objective_sense="max",
    objective_func=objective,
    solution_length=GENOME_LENGTH,
    device=DEVICE,
    vectorized=True,
)

searcher = PGPE(
    problem,
    popsize=SEARCH_POPSIZE,
    center_learning_rate=CENTER_LEARNING_RATE,
    stdev_learning_rate=STDEV_LEARNING_RATE,
    stdev_init=STDEV_INIT,
    center_init=center_init,
    optimizer="clipup",
    optimizer_config={"max_speed": MAX_SPEED},
)

# ==== 5. RUN + SHOW RESULTS ===================================================
StdOutLogger(searcher)
searcher.run(NUM_GENERATIONS)

print("\nFinal searcher status:")
print(searcher.status)
