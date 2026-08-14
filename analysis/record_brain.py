"""Lets a run's BEST genome live through its own paradigm again -- as ONE individual
(pop=1), with fresh random initial weights just like a real evolutionary evaluation --
so every tick can be watched instead of only judged by its final fitness.

This drives sim_core.maze_task.simulate_training_phase / sim_core.replay_task.simulate_replay_phase
directly -- the exact functions the evolutionary search itself calls -- with an
optional per-tick `recorder` hook added to simulate_training_phase purely as an
opt-in callback (see BrainRecorder below). fitness.py's real evolutionary calls
never pass a recorder, so their behavior/RNG consumption is completely unchanged
by this script's existence.

Since a genuinely robust solver should behave the same way across evaluations, this
script does not replay the original run's exact RNG seeds -- it draws its own fresh
ones (see RECORDING_*_SEED below) rather than needing bit-identical reproduction of
any specific evolutionary generation.

Only trainA/trainB phases are recorded/plotted, for the first and last few maze runs
of EACH such phase in the paradigm (numbered maze0, maze1, ... in the order they
appear, replay phases don't get a number). Replay phases have no "maze run" concept
(see paradigm.py) so they're simulated normally to keep the CTRNN state/weights
chain intact, just not visualized here.

This script never writes to the h5 file or any other run output -- read-only in,
new plots out.

Usage: python -m analysis.record_brain <path_to_run_dir> <device>
"""

# ==== 1) IMPORTS =================================================================
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")  # headless -- this only ever calls savefig(), never plt.show(), and a
                        # remote GPU server reached over VPN may have no display/Tk at all
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.results_io import results_filename
from sim_core import constants
from sim_core.genome_codec import sample_initial_weights
from sim_core.maze_task import simulate_training_phase
from sim_core.paradigm import PHASE_REPLAY, PHASE_TRAIN_A, PHASE_TRAIN_B, parse_paradigm
from sim_core.replay_task import simulate_replay_phase

# ==== 2) CONSTANTS / USER INPUTS =================================================
PLOT_DPI = 180
HEATMAP_CMAP = "coolwarm"  # diverging, centered at 0 -- matches analysis/view_genome.py

N_RUNS_TO_RECORD_HEAD = 5  # first N maze runs of every trainA/trainB phase to plot
N_RUNS_TO_RECORD_TAIL = 5  # last N maze runs of every trainA/trainB phase to plot -- overlaps
                            # with the head set are just plotted once (see BrainRecorder.begin_phase)

# fresh RNG seeds for this diagnostic re-evaluation -- deliberately NOT the original
# run's seeds, see module docstring. Three separate streams, same separation-of-concerns
# as run_evolution.py's noise/reward/weight-init generators.
RECORDING_NOISE_SEED = 101
RECORDING_REWARD_SEED = 202
RECORDING_WEIGHT_INIT_SEED = 303

# tick position -> human label within one 7-tick maze run (fixed/structural, see
# constants.py's TICKS_PER_RUN docstring: 1=home, 2/3/5/6=corridor, 4=turn, 7=mazeend)
TICK_LABELS = {1: "home", 2: "corridor", 3: "corridor", 4: "turn", 5: "corridor", 6: "corridor", 7: "mazeend"}

PHASE_CONTEXT = {PHASE_TRAIN_A: "A", PHASE_TRAIN_B: "B"}

if len(sys.argv) != 3:
    raise ValueError("Usage: python -m analysis.record_brain <path_to_run_dir> <device>")
RUN_DIR = Path(sys.argv[1])
DEVICE = sys.argv[2]
STEM = RUN_DIR.name
H5_PATH = RUN_DIR / results_filename(STEM)
OUTPUT_DIR = RUN_DIR


# ==== 3) CONFIG + BEST GENOME LOADING (READ-ONLY) ================================
def _load_config_and_best_genome(h5_path):
    with h5py.File(h5_path, "r") as h5_file:
        config = json.loads(h5_file["meta"].attrs["config_json"])
        best_group = h5_file["genome/best"]
        genome = {name: torch.tensor(best_group[name][()][0]) for name in ("A", "B", "C", "D", "eta", "beta", "M")}
        fitness = float(best_group["fitness"][()][0])
    return config, genome, fitness


# ==== 4) DECISION LABEL ===========================================================
def _decision_label(output_value):
    """Same threshold rule as maze_task.py's arm_choice logic."""
    if output_value >= constants.STRAIGHT_THRESH:
        return "right"
    if output_value <= -constants.STRAIGHT_THRESH:
        return "left"
    return "straight"


# ==== 5) CUE READOUT ==============================================================
def _cue_summary(ticks, phase_type):
    """Context/sensory cue values are clamped input-neuron activations, constant for
    the whole run (see maze_task.py's _initialize_training_phase), so any tick's state
    works -- reads tick 0's. Also derives the actually-correct turn (the one the
    reward schedule pays out for) from the cue + context rule (context A = direct
    mapping, context B = mirror -- see maze_task.py's
    _context_transform_arm/_sensory_from_arm), so it can be eyeballed against the
    network's own decision above without re-deriving the mapping by hand each time.
    This is what's correct given the cue, not a prediction of what the network will do
    -- an untrained/frozen network has no reason to match it.

    Falls back to 'ablated/off' when a cue channel reads as all-zero, which only
    happens when evo_context/sensory_cues_on was False for this run -- in that case
    the network never saw the cue, and the true generated cue value isn't recoverable
    from clamped state either, so the correct turn can't be determined."""
    sample_state = ticks[0]["state"]
    sensory_a_on = sample_state[constants.INPUT_SENSORY_A] > 0.5
    sensory_b_on = sample_state[constants.INPUT_SENSORY_B] > 0.5
    context_a_on = sample_state[constants.INPUT_CONTEXT_A] > 0.5
    context_b_on = sample_state[constants.INPUT_CONTEXT_B] > 0.5

    context_label = "A" if context_a_on else ("B" if context_b_on else "ablated/off")

    if not sensory_a_on and not sensory_b_on:
        return "ablated/off", context_label, "n/a (sensory cue ablated)"

    sensory_label = "A" if sensory_a_on else "B"
    cue_arm = 0 if sensory_a_on else 1  # cue_arm=0 <-> sensory A, see _sensory_from_arm
    correct_arm = cue_arm if phase_type == PHASE_TRAIN_A else 1 - cue_arm  # context B mirrors it
    correct_label = "left" if correct_arm == 0 else "right"
    return sensory_label, context_label, correct_label


# ==== 6) PER-RUN FACET PLOT =======================================================
def _plot_run(stem, output_dir, maze_idx, phase_type, run_idx, ticks):
    """ticks: list of per-tick dicts (step_in_run, state, W, output, crashed, rewarded),
    in temporal order, for one maze run. Up to 7 columns, one per tick, plus one joint
    text row (spans all columns -- these values are constant for the whole run)."""
    n_ticks = len(ticks)
    n_neurons = ticks[0]["state"].shape[0]
    neuron_ids = np.arange(n_neurons)

    # equal-width columns across every row (no per-row colorbar eating into one row's
    # width, which is what threw the columns out of alignment) -- the colorbar is
    # instead one full-width horizontal axis at the bottom, next to the cue-summary row.
    # One shared scale for both W and activation, always exactly [-1, 1]: W is
    # renormalized to max-abs-1 every tick (see maze_task.py's plasticity_step call
    # site), and activation is a state/tanh(...) blend that stays within [-1, 1] as
    # long as dt/tau <= 1 (true for every config so far).
    figure = plt.figure(figsize=(2.6 * n_ticks, 9.7), dpi=PLOT_DPI)
    grid = figure.add_gridspec(nrows=5, ncols=n_ticks, height_ratios=[4, 1, 1, 0.3, 0.6], hspace=0.9)

    im = None
    for col, tick in enumerate(ticks):
        w_axis = figure.add_subplot(grid[0, col])
        im = w_axis.imshow(tick["W"], cmap=HEATMAP_CMAP, vmin=-1.0, vmax=1.0, aspect="equal")
        w_axis.set_title(f"tick {tick['step_in_run']} ({TICK_LABELS[tick['step_in_run']]})", fontsize=9)
        w_axis.set_xticks(neuron_ids)
        w_axis.set_yticks(neuron_ids)
        w_axis.tick_params(labelsize=6)
        w_axis.set_xlabel("pre neuron (j)", fontsize=8)
        if col == 0:
            w_axis.set_ylabel("post neuron (i)", fontsize=8)

        activation_axis = figure.add_subplot(grid[1, col])
        activation_axis.imshow(tick["state"][None, :], cmap=HEATMAP_CMAP, vmin=-1.0, vmax=1.0, aspect="auto")
        activation_axis.set_xticks(neuron_ids)
        activation_axis.set_yticks([])
        activation_axis.tick_params(labelsize=6)
        activation_axis.set_xlabel("neuron ID", fontsize=8)

        text_axis = figure.add_subplot(grid[2, col])
        text_axis.axis("off")
        text_axis.text(0.5, 0.5, f"{tick['output']:.2f} -> {_decision_label(tick['output'])}", ha="center", va="center", fontsize=9)

    figure.colorbar(im, cax=figure.add_subplot(grid[3, :]), orientation="horizontal", label="W / activation")

    sensory_label, context_label, correct_label = _cue_summary(ticks, phase_type)
    cue_axis = figure.add_subplot(grid[4, :])
    cue_axis.axis("off")
    cue_axis.text(
        0.5, 0.5, f"context: {context_label}  |  sensory cue: {sensory_label}  |  correct turn: {correct_label}",
        ha="center", va="center", fontsize=10,
    )

    turn_tick = next((tick for tick in ticks if tick["step_in_run"] == 4), None)
    decision_text = _decision_label(turn_tick["output"]) if turn_tick is not None else "never reached turn tick"
    outcome = "crashed" if ticks[-1]["crashed"] else ("rewarded (reached mazeend)" if ticks[-1]["rewarded"] else "ongoing")
    figure.suptitle(f"{stem} -- maze{maze_idx} {phase_type} run {run_idx} -- decision: {decision_text} -- {outcome}")

    output_path = output_dir / f"{stem}_maze{maze_idx}_{phase_type}_run{run_idx}.png"
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


# ==== 6) RECORDER (opt-in hook passed into simulate_training_phase) ==============
class BrainRecorder:
    """Buffers per-tick data for the head/tail maze runs of interest in the CURRENT
    phase, then renders one facet plot per recorded run when that phase ends. Assumes
    pop=1 (this script's only use case) -- batched tensors passed in are always
    indexed at [0]."""

    def __init__(self, stem, output_dir):
        self._stem = stem
        self._output_dir = output_dir
        self._maze_idx = None
        self._phase_type = None
        self._ticks_by_run = {}

    def begin_phase(self, maze_idx, phase_type, num_runs):
        self._maze_idx = maze_idx
        self._phase_type = phase_type
        head = set(range(min(N_RUNS_TO_RECORD_HEAD, num_runs)))
        tail = set(range(max(0, num_runs - N_RUNS_TO_RECORD_TAIL), num_runs))
        self._ticks_by_run = {run_idx: [] for run_idx in (head | tail)}

    def on_training_tick(self, run_index, step_in_run, state, W, output, crashed, rewarded):
        run_idx = int(run_index[0].item())
        if run_idx not in self._ticks_by_run:
            return
        self._ticks_by_run[run_idx].append({
            "step_in_run": int(step_in_run[0].item()),
            "state": state[0].detach().cpu().numpy(),
            "W": W[0].detach().cpu().numpy(),
            "output": float(output[0].item()),
            "crashed": bool(crashed[0].item()),
            "rewarded": bool(rewarded[0].item()),
        })

    def end_phase(self):
        for run_idx in sorted(self._ticks_by_run):
            ticks = self._ticks_by_run[run_idx]
            if not ticks:
                continue
            _plot_run(self._stem, self._output_dir, self._maze_idx, self._phase_type, run_idx, ticks)
        self._ticks_by_run = {}


# ==== 7) MAIN EXECUTION ===========================================================
CONFIG, GENOME, BEST_FITNESS = _load_config_and_best_genome(H5_PATH)

constants.configure_network(n_neurons=CONFIG["n_neurons"])
constants.configure(
    dt=CONFIG["dt"], tau=CONFIG["tau"], noise_std=CONFIG["noise_std"],
    straight_thresh=CONFIG["straight_thresh"], big_reward=CONFIG["big_reward"],
    small_reward=CONFIG["small_reward"], crash_penalty=CONFIG["crash_penalty"],
    turn_reward_big=CONFIG["turn_reward_big"], turn_reward_small=CONFIG["turn_reward_small"],
)

# replicate fitness.py's evaluate_generation exactly: eta is forced to zero when
# plasticity was ablated during evolution, regardless of whatever the saved genome's
# raw eta values are (see sim_core/fitness.py's evaluate_generation)
if not CONFIG["evo_plasticity_on"]:
    GENOME["eta"] = torch.zeros_like(GENOME["eta"])

PARADIGM_PHASES = parse_paradigm(CONFIG["paradigm"])
GENOME_BATCHED = {name: tensor.unsqueeze(0).to(DEVICE) for name, tensor in GENOME.items()}

noise_generator = torch.Generator(device=DEVICE)
noise_generator.manual_seed(RECORDING_NOISE_SEED)
reward_generator = torch.Generator(device=DEVICE)
reward_generator.manual_seed(RECORDING_REWARD_SEED)
weight_init_generator = torch.Generator(device=DEVICE)
weight_init_generator.manual_seed(RECORDING_WEIGHT_INIT_SEED)

state = torch.zeros(1, constants.N, device=DEVICE)
W = sample_initial_weights(1, DEVICE, weight_init_generator)

recorder = BrainRecorder(STEM, OUTPUT_DIR)
train_phase_idx = 0
for phase_type, value in PARADIGM_PHASES:
    if phase_type in PHASE_CONTEXT:
        recorder.begin_phase(train_phase_idx, phase_type, value)
        state, W, _ = simulate_training_phase(
            state, W, GENOME_BATCHED["M"], GENOME_BATCHED["A"], GENOME_BATCHED["B"],
            GENOME_BATCHED["C"], GENOME_BATCHED["D"], GENOME_BATCHED["beta"], GENOME_BATCHED["eta"],
            PHASE_CONTEXT[phase_type], value,
            CONFIG["evo_context_cues_on"], CONFIG["evo_sensory_cues_on"],
            noise_generator, reward_generator, DEVICE,
            recorder=recorder,
        )
        recorder.end_phase()
        train_phase_idx += 1
    elif phase_type == PHASE_REPLAY:
        state, W, _ = simulate_replay_phase(
            state, W, GENOME_BATCHED["M"], GENOME_BATCHED["A"], GENOME_BATCHED["B"],
            GENOME_BATCHED["C"], GENOME_BATCHED["D"], GENOME_BATCHED["beta"], GENOME_BATCHED["eta"],
            value, noise_generator, DEVICE,
        )
    else:
        raise ValueError(f"Unknown phase type '{phase_type}' in paradigm.")

print(f"Recorded {train_phase_idx} maze phase(s) (fitness={BEST_FITNESS:.4f}) for {STEM} to: {OUTPUT_DIR}")
