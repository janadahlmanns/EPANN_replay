"""First raw end-to-end run: EvoTorch PGPE search over the flattened CTRNN genome,
using a random genome as the search center, for a handful of generations, printing results.
Not tuned/validated yet -- this is the "get it running"
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

from sim_core.genome_codec import GENOME_LENGTH
from sim_core.fitness import fitness_function

# ==== 2. CONSTANTS / USER INPUTS =============================================
DEVICE = "cuda"
MASTER_SEED = 0
NOISE_SEED = 1
REWARD_SEED = 2

NUM_GENERATIONS = 200
SEARCH_POPSIZE = 200           
RADIUS_INIT = 50.0            # radius of the initial search hypersphere in genome space (GENOME_LENGTH-dim), sweep/ optimize
MAX_SPEED = RADIUS_INIT / 15  # evotorch's rule of thumb from the ClipUp paper: max_speed = radius / 15.0, adjust the 15.0 to optimize
CENTER_LEARNING_RATE = MAX_SPEED / 2  # this is the step size in the ClipUp paper
STDEV_LEARNING_RATE = 0.1
MOMENTUM = 0.9  

L1_LAMBDA = 1e-3

torch.manual_seed(MASTER_SEED)
torch.cuda.manual_seed_all(MASTER_SEED)
torch.use_deterministic_algorithms(True)

noise_generator = torch.Generator(device=DEVICE)
noise_generator.manual_seed(NOISE_SEED)
reward_generator = torch.Generator(device=DEVICE)
reward_generator.manual_seed(REWARD_SEED)

# ==== 3. BUILD CENTER_INIT FROM A RANDOM GENOME VECTOR =======================
center_init = torch.randn(GENOME_LENGTH, device=DEVICE)

# ==== 4. BUILD EVOTORCH PROBLEM + PGPE SEARCHER ==============================
objective = functools.partial(
    fitness_function,
    device=DEVICE,
    noise_generator=noise_generator,
    reward_generator=reward_generator,
    l1_lambda=L1_LAMBDA,
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
    radius_init=RADIUS_INIT,
    center_init=center_init,
    optimizer="clipup",
    optimizer_config={"max_speed": MAX_SPEED, "momentum": MOMENTUM},
)

# ==== 5. RUN + SHOW RESULTS ===================================================
StdOutLogger(searcher)
searcher.run(NUM_GENERATIONS)

print("\nFinal searcher status:")
print(searcher.status)
