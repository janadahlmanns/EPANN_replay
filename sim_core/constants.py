"""Shared constants for the CTRNN T-maze simulation core. Imported by every sim_core module."""

# ==== NETWORK LAYOUT =========================================================
# neuron index mapping: [0..6] = inputs, [7] = output, [8..19] = free hidden units
N = 20
N_INPUT = 7
OUTPUT_IDX = 7
HIDDEN_START = 8

# input channel order within the first N_INPUT neurons
INPUT_HOME = 0
INPUT_TURN = 1
INPUT_MAZEEND = 2
INPUT_CONTEXT_A = 3
INPUT_CONTEXT_B = 4
INPUT_SENSORY_A = 5
INPUT_SENSORY_B = 6

# ==== CTRNN DYNAMICS ==========================================================
DT = 0.2
TAU = 1.0
NOISE_STD = 0.1

# ==== MAZE TASK ===============================================================
STRAIGHT_THRESH = 1.0 / 3.0
BIG_REWARD = 1.0
SMALL_REWARD = 0.2 # !!changed from 0.2 as in Soltoggio et al. to try something
CRASH_PENALTY = -0.4
NUM_RUNS_PER_TRAINING_PHASE = 100
TICKS_PER_RUN = 7                      # 1,2,3=straight 4=turn 5,6,7=straight(7=mazeend)
MAX_TRAINING_TICKS = NUM_RUNS_PER_TRAINING_PHASE * TICKS_PER_RUN

# arm / sensory-cue / turn-sign mapping:
#   arm 0 <-> sensory cue "a" <-> turn output <= -STRAIGHT_THRESH
#   arm 1 <-> sensory cue "b" <-> turn output >=  STRAIGHT_THRESH

# ==== REPLAY PHASE =============================================================
REPLAY_TICKS = 10
