"""Runs run_evolution.py once per config file in a batch folder, each as its own
subprocess (clean isolation -- no shared CUDA/RNG/module state between runs). Each
config's own "run_name" (i.e. its filename) becomes its output plot folder, per
run_evolution.py's own convention.
"""

import subprocess
import sys
from pathlib import Path

# ==== 1) CONSTANTS / USER INPUTS ================================================
BATCH_FOLDER = Path("C:/EPANN_replay/runners/batches_to_run") # put all input json to be run into this folder, script then runs all consecutively
RUN_EVOLUTION_SCRIPT = Path("C:/EPANN_replay/runners/run_evolution.py")


# ==== 2) BATCH EXECUTION =========================================================
config_paths = sorted(BATCH_FOLDER.glob("*.json"))
if not config_paths:
    raise ValueError(f"No .json config files found in {BATCH_FOLDER}")

for idx, config_path in enumerate(config_paths, start=1):
    print(f"\n{'=' * 90}\nBATCH RUN {idx}/{len(config_paths)}: {config_path.name}\n{'=' * 90}\n")
    subprocess.run([sys.executable, str(RUN_EVOLUTION_SCRIPT), str(config_path)], check=True)

print(f"\nBatch complete: {len(config_paths)} runs from {BATCH_FOLDER}")
