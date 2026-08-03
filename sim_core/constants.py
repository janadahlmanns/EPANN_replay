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

# ==== TUNABLE SIMULATION CONSTANTS ============================================
# These are per-run experiment parameters, not fixed structure: they start as
# None and MUST be set via configure() (called once from the runner, from its
# loaded config file) before any sim_core module reads them. Other sim_core
# modules import the `constants` module itself (not these names directly) and
# read e.g. `constants.NOISE_STD` at call time, so configure() only needs to run
# before the simulation actually executes, not before these modules are imported.
DT = None
TAU = None
NOISE_STD = None
STRAIGHT_THRESH = None
BIG_REWARD = None
SMALL_REWARD = None
CRASH_PENALTY = None
TURN_REWARD_BIG = None
TURN_REWARD_SMALL = None

TICKS_PER_RUN = 7                      # fixed: maze_task.py's turn/end-tick checks are
                                        # hardcoded to this length (1,2,3=straight 4=turn
                                        # 5,6,7=straight(7=mazeend)), so this is structural,
                                        # not a tunable experiment parameter


def configure(dt, tau, noise_std, straight_thresh, big_reward, small_reward,
              crash_penalty, turn_reward_big, turn_reward_small):
    """Set all tunable simulation constants for this run (called once by the runner)."""
    global DT, TAU, NOISE_STD, STRAIGHT_THRESH, BIG_REWARD, SMALL_REWARD
    global CRASH_PENALTY, TURN_REWARD_BIG, TURN_REWARD_SMALL
    DT = dt
    TAU = tau
    NOISE_STD = noise_std
    STRAIGHT_THRESH = straight_thresh
    BIG_REWARD = big_reward
    SMALL_REWARD = small_reward
    CRASH_PENALTY = crash_penalty
    TURN_REWARD_BIG = turn_reward_big
    TURN_REWARD_SMALL = turn_reward_small

# arm / sensory-cue / turn-sign mapping:
#   arm 0 <-> sensory cue "a" <-> turn output <= -STRAIGHT_THRESH
#   arm 1 <-> sensory cue "b" <-> turn output >=  STRAIGHT_THRESH

# ==== REWARD SCHEDULE ==========================================================
# A run pays out in up to two installments, both using the SAME per-arm
# magnitude (BIG_REWARD if the arm chosen at the turn matches this run's
# big-reward arm, else SMALL_REWARD):
#   1) at the turn tick, immediately upon picking a valid direction (does not
#      end the run -- the agent keeps going down the chosen corridor)
#   2) at the mazeend tick, if the agent reaches it without crashing
#
# A crash always costs CRASH_PENALTY and ends the run immediately. Any turn
# payout already earned this run is kept (it is not clawed back), so:
#   crash before/at the turn (never turned)      -> CRASH_PENALTY
#   crash after a correct-arm turn                -> TURN_REWARD_BIG + CRASH_PENALTY
#   crash after a wrong-arm turn                   -> TURN_REWARD_SMALL + CRASH_PENALTY
#   full run, correct arm (max payout)             -> TURN_REWARD_BIG + BIG_REWARD
#   full run, wrong arm                            -> TURN_REWARD_SMALL + SMALL_REWARD
