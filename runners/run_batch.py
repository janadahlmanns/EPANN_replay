"""Runs run_evolution.py once per config file in a batch folder, each as its own
subprocess (clean isolation -- no shared CUDA/RNG/module state between runs). Each
config's own "run_name" (i.e. its filename) becomes its output plot folder, per
run_evolution.py's own convention.

RUN_IN_PARALLEL controls how many of these subprocess chains run at once (each
chain still runs its own configs one after another). Only raise this above 1 when
you know the GPU has headroom for it -- e.g. running 2 side by side, like manually
running run_batch.py in two terminals on two halves of the folder.
"""

import subprocess
import sys
import threading
from pathlib import Path

# ==== 1) CONSTANTS / USER INPUTS ================================================
BATCH_FOLDER = Path("C:/EPANN_replay/runners/batches_to_run") # put all input json to be run into this folder, script then runs all consecutively
RUN_EVOLUTION_SCRIPT = Path("C:/EPANN_replay/runners/run_evolution.py")
RUN_IN_PARALLEL = 2  # how many config chains to run concurrently -- only raise this if you're at
                     # the computer and sure the sims haven't grown enough to fight over GPU memory


# ==== 2) BATCH EXECUTION =========================================================
def _run_chain(chain_idx, chain_config_paths):
    for config_path in chain_config_paths:
        print(f"\n{'=' * 90}\nCHAIN {chain_idx}: {config_path.name}\n{'=' * 90}\n")
        subprocess.run([sys.executable, str(RUN_EVOLUTION_SCRIPT), str(config_path)], check=True)


config_paths = sorted(BATCH_FOLDER.glob("*.json"))
if not config_paths:
    raise ValueError(f"No .json config files found in {BATCH_FOLDER}")

# Warm up imports of unsigned/native-extension libs (matplotlib, torch, ...) in a
# throwaway subprocess *before* spawning the parallel chains. Without this, two chains
# can both import them for the first time at the same instant, and Windows'
# application-control/reputation check on the never-before-seen DLLs can flake and
# block one of the concurrent loads (seen with matplotlib's _image DLL).
subprocess.run([sys.executable, "-c", "import matplotlib.pyplot, torch"], check=True)

# split into RUN_IN_PARALLEL contiguous chunks, one chain per chunk
chains = [config_paths[chain_idx::RUN_IN_PARALLEL] for chain_idx in range(RUN_IN_PARALLEL)]

threads = [
    threading.Thread(target=_run_chain, args=(chain_idx, chain_config_paths))
    for chain_idx, chain_config_paths in enumerate(chains)
    if chain_config_paths
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

print(f"\nBatch complete: {len(config_paths)} runs from {BATCH_FOLDER} across {len(threads)} parallel chain(s)")
