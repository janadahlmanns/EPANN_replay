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


# ==== 3) VARIANTS (per-individual paradigm draw) ================================
def parse_paradigm_variants(paradigm_field):
    """Config's "paradigm" field -> list of parsed phase-lists.

    Accepts either a single paradigm string (every existing config; behaves exactly as
    before -- returns a 1-element list) or a JSON list of paradigm strings (multiple
    equally-likely variants: run_evolution.py draws one independently per individual,
    per generation -- see fitness.py's evaluate_generation/_run_paradigm_per_individual).
    There is no explicit weighting -- N variants means each gets probability 1/N.

    All variants must have IDENTICAL shape: the same number of phases, the same
    (run/tick count) VALUE at each position, AND the same CATEGORY (training vs.
    replay) at each position -- only WHICH training task (trainA vs trainB) may differ
    position by position (e.g. "trainA, 100" vs "trainB, 100"). This is what lets
    fitness.py's _run_paradigm_per_individual process the whole population through one
    shared tick loop per POSITION (never one loop per variant -- see its docstring):
    every downstream tensor (tracking, plots) keeps one fixed shape regardless of which
    variant an individual drew, and a replay position never needs to be mixed with a
    training one within the same population-wide call."""
    if isinstance(paradigm_field, str):
        return [parse_paradigm(paradigm_field)]
    if not isinstance(paradigm_field, list) or not paradigm_field:
        raise ValueError(f"paradigm must be a string or a non-empty list of strings, got {paradigm_field!r}")

    variants = [parse_paradigm(variant_str) for variant_str in paradigm_field]
    reference_shape = [(value, phase_type == PHASE_REPLAY) for phase_type, value in variants[0]]
    for variant_str, variant_phases in zip(paradigm_field, variants):
        shape = [(value, phase_type == PHASE_REPLAY) for phase_type, value in variant_phases]
        if shape != reference_shape:
            raise ValueError(
                "All paradigm variants must have the same shape (same number of phases, same "
                "run/tick count AND same training-vs-replay category at each position); variant "
                f"{variant_str!r} has (value, is_replay) sequence {shape}, expected {reference_shape} "
                f"(from {paradigm_field[0]!r})."
            )
    return variants
