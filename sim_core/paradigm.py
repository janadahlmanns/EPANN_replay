"""Parses a paradigm string into an ordered list of training/replay phases.

Shared by run_evolution.py and (later) run_test.py so both runners describe
their per-evaluation phase sequence the same way.
"""

# ==== 1) CONSTANTS ==============================================================
PHASE_TRAIN_A = "trainA"
PHASE_TRAIN_B = "trainB"
PHASE_REPLAY = "replay"
VALID_PHASE_TYPES = (PHASE_TRAIN_A, PHASE_TRAIN_B, PHASE_REPLAY)


# ==== 2) PARSER ==================================================================
def parse_paradigm(paradigm_str):
    """Parse e.g. "trainA, 100, replay, 10, trainB, 50, replay, 30" into
    [("trainA", 100), ("replay", 10), ("trainB", 50), ("replay", 30)].

    Phase types must be one of VALID_PHASE_TYPES; their paired value must be a
    positive integer (maze-run count for trainA/trainB, tick count for replay).
    Anything else raises ValueError.
    """
    tokens = [token.strip() for token in paradigm_str.split(",")]
    if len(tokens) % 2 != 0:
        raise ValueError(f"Paradigm string must have an even number of tokens (phase, value pairs): {paradigm_str!r}")

    phases = []
    for phase_type, value_str in zip(tokens[0::2], tokens[1::2]):
        if phase_type not in VALID_PHASE_TYPES:
            raise ValueError(f"Invalid phase type {phase_type!r} in paradigm string; valid types are {VALID_PHASE_TYPES}")
        value = int(value_str)
        if value <= 0:
            raise ValueError(f"Phase {phase_type!r} value must be a positive integer, got {value_str!r}")
        phases.append((phase_type, value))
    return phases
