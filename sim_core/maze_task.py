"""Batched T-maze training phase (Soltoggio-style T-maze/reversal paradigm).

Each individual in the batch runs its own independent sequence of maze runs, all
stepped in lockstep tick-by-tick so the whole population stays vectorized on GPU.
A run is 7 ticks (home, straight, straight, turn, straight, straight, mazeend) if
never crashed; a wrong output at any tick ends the run immediately (crash penalty,
reset to start). The CTRNN state and weights W are NEVER reset between runs or
maze resets -- only the maze/task bookkeeping (position, reward arm, sensory cue)
resets. This is what makes the plasticity itself the thing being evolved.

context_cues_on / sensory_cues_on: when False, the corresponding input neurons are
clipped to zero activity every tick instead of carrying their normal cue value. The
neurons stay in the network (same neuron count in every condition) -- they are just
denied any signal, so evolution cannot repurpose them as free processing units.

Reward schedule (see constants.py "TUNABLE SIMULATION CONSTANTS" section for the full spec):
  - Crashing (choosing straight when a turn is required, or vice versa) always
    costs CRASH_PENALTY and ends the run immediately, no matter which tick it
    happens on.
  - Choosing a valid direction at the turn tick pays out immediately (does NOT
    end the run) -- TURN_REWARD_BIG if that arm matches this run's big-reward
    arm, else TURN_REWARD_SMALL. The run then continues down that corridor.
  - Reaching mazeend without crashing pays out again, using the same
    big/small-arm logic, ending the run.
  - Any turn payout already earned is kept even if the run later crashes, so a
    fully successful run pays out twice (up to BIG_REWARD + BIG_REWARD = 2.0),
    a run that crashes after turning keeps the turn payout minus CRASH_PENALTY,
    and a run that crashes before ever turning only pays CRASH_PENALTY.
"""

import torch
from sim_core import constants
from sim_core.constants import N_INPUT, OUTPUT_IDX, TICKS_PER_RUN
from sim_core.ctrnn import activation_step, plasticity_step


# ==== ONLINE REWARD SIGNAL (spec 3.11) ========================================
# The reward input neuron carries the reward/penalty earned by the PREVIOUS tick's
# action -- the tick that earned it can't feed it into its own input vector because
# that tick's output is already produced before its outcome is known. So a reward
# earned at tick t is presented as the input at tick t+1, then goes back to 0 at
# t+2. This is unconditional (unlike context/sensory cues, it isn't gated by any
# on/off flag) and separate from `total_reward`/evolutionary fitness -- callers
# that need reward WITHOUT it counting toward fitness (e.g. a future unrewarded
# exploration phase) still get it wired into the network here; they just don't
# feed this function's total_reward into the fitness computation.


# ==== HELPERS =================================================================
def _draw_reward_arm(pop, reward_generator, device):
    return torch.randint(0, 2, (pop,), generator=reward_generator, device=device)


def _context_transform_arm(big_reward_arm, context):
    """Map the rewarded arm to the sensory-cue arm under the given context's rule.

    context "A" = direct mapping (cue points straight at the reward).
    context "B" = lateral/mirror mapping (cue points at the reflected side).
    In this single (2-arm) T-maze, "mirror" and "the other arm" happen to be the
    same operation -- but they will NOT be the same once the double T-maze adds
    context C (diagonal mapping), so this function -- not a generic "1 - arm"
    inline -- is the one place that must grow a "C" branch later.
    """
    if context == "A":
        return big_reward_arm
    if context == "B":
        return 1 - big_reward_arm
    raise ValueError(f"Unknown context '{context}'; valid contexts are 'A', 'B'.")


def _sensory_from_arm(big_reward_arm, context):
    """context: python str ('A' or 'B'), same context for the whole batch this call."""
    cue_arm = _context_transform_arm(big_reward_arm, context)
    sensory_a = (cue_arm == 0).float()
    sensory_b = 1.0 - sensory_a
    return sensory_a, sensory_b


def _initialize_training_phase(state, context, reward_generator, device):
    pop = state.shape[0]
    state = state.clone()

    step_in_run = torch.ones(pop, dtype=torch.long, device=device)
    run_count = torch.zeros(pop, dtype=torch.long, device=device)
    chosen_arm = torch.zeros(pop, dtype=torch.long, device=device)
    total_reward = torch.zeros(pop, device=device)

    context_a = torch.full((pop,), 1.0 if context == "A" else 0.0, device=device)
    context_b = torch.full((pop,), 1.0 if context == "B" else 0.0, device=device)

    big_reward_arm = _draw_reward_arm(pop, reward_generator, device)
    sensory_a, sensory_b = _sensory_from_arm(big_reward_arm, context)

    return (
        pop,
        state,
        step_in_run,
        run_count,
        chosen_arm,
        total_reward,
        context_a,
        context_b,
        big_reward_arm,
        sensory_a,
        sensory_b,
    )


# ==== MAIN SIMULATION ==========================================================
def simulate_training_phase(state, W, M, A, B, C, D, beta, eta,
                             context, num_runs, context_cues_on, sensory_cues_on,
                             noise_generator, reward_generator, device,
                             collect_tracking=False, recorder=None):
    """Runs num_runs maze runs (batched over population).

    recorder: optional per-tick recording hook (see analysis/record_brain.py's
    BrainRecorder) for watching a single individual live through its evaluation.
    When None (always the case for fitness.py's real evolutionary calls), this
    function's behavior and RNG consumption are completely unchanged -- the only
    added cost is one `is None` check per tick. Batched tensors are passed to it
    as-is (recorder itself owns any pop-indexing it needs)."""
    max_ticks = num_runs * TICKS_PER_RUN
    (
        pop,
        state,
        step_in_run,
        run_count,
        chosen_arm,
        total_reward,
        context_a,
        context_b,
        big_reward_arm,
        sensory_a,
        sensory_b,
    ) = _initialize_training_phase(state, context, reward_generator, device)

    recent_reward = torch.zeros(pop, device=device)  # reward earned last tick; fed as this tick's input, see module docstring

    tracking = None
    current_turn_choice = None
    if collect_tracking:
        current_turn_choice = torch.full((pop,), -1, dtype=torch.long, device=device)
        tracking = {
            # -1 = no turn recorded this run, 0 = left, 1 = right
            "decisions_by_run": torch.full((pop, num_runs), -1, dtype=torch.long, device=device),
            "crashed_by_run": torch.zeros((pop, num_runs), dtype=torch.bool, device=device),
            "rewarded_by_run": torch.zeros((pop, num_runs), dtype=torch.bool, device=device),
            # kept for backward compatibility -- equals correct_arm_by_run whenever rewarded_by_run is True
            "big_reward_by_run": torch.zeros((pop, num_runs), dtype=torch.bool, device=device),
            "sensory_cue_by_run": torch.zeros((pop, num_runs), dtype=torch.long, device=device),
            # was the direction chosen at the turn tick the run's big-reward arm? only meaningful
            # where decisions_by_run != -1 (i.e. a turn actually happened this run)
            "correct_arm_by_run": torch.zeros((pop, num_runs), dtype=torch.bool, device=device),
        }
        tracking["sensory_cue_by_run"][:, 0] = (sensory_b > 0.5).long()

    for _ in range(max_ticks):
        active = run_count < num_runs
        if not torch.any(active):
            break # break this generation's loop because all individuals allready finished their runs

        # --- build this tick's input vector ---
        is_home = (step_in_run == 1).float() # gives 1 for home and 0 otherwise
        is_turn_tick = step_in_run == 4 # gives True for the turn tick and False otherwise
        is_end_tick = step_in_run == 7 # gives True for mazeend and False otherwise

        # clip context/sensory neurons to zero activity when their cue is toggled off,
        # rather than dropping the neurons, so neuron count stays fixed across conditions
        input_context_a = context_a if context_cues_on else torch.zeros_like(context_a)
        input_context_b = context_b if context_cues_on else torch.zeros_like(context_b)
        input_sensory_a = sensory_a if sensory_cues_on else torch.zeros_like(sensory_a)
        input_sensory_b = sensory_b if sensory_cues_on else torch.zeros_like(sensory_b)
        input_vec = torch.stack(
            [is_home, is_turn_tick.float(), is_end_tick.float(),
             input_context_a, input_context_b, input_sensory_a, input_sensory_b,
             recent_reward], dim=1,
        )

        # --- CTRNN tick: clamp inputs, advance state, apply plasticity ---
        state[:, :N_INPUT] = input_vec
        new_state = activation_step(state, W, beta, constants.NOISE_STD, noise_generator)
        dW = plasticity_step(state, W, M, A, B, C, D, eta)
        W = W + dW
        W = W / W.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        new_state[:, :N_INPUT] = input_vec
        output = new_state[:, OUTPUT_IDX]

        # --- score this tick ---
        straight_ok = output.abs() < constants.STRAIGHT_THRESH
        turn_ok = output.abs() >= constants.STRAIGHT_THRESH
        correct = torch.where(is_turn_tick, turn_ok, straight_ok)

        arm_choice = torch.where(
            output >= constants.STRAIGHT_THRESH,
            torch.ones_like(chosen_arm),
            torch.where(
                output <= -constants.STRAIGHT_THRESH,
                torch.zeros_like(chosen_arm),
                torch.full_like(chosen_arm, -1),
            ),
        )
        chosen_arm = torch.where(is_turn_tick, arm_choice, chosen_arm)

        crash = active & (~correct)
        got_reward = active & is_end_tick & correct
        turned_correctly = active & is_turn_tick & correct  # valid direction chosen; run continues

        if recorder is not None:
            recorder.on_training_tick(
                run_index=run_count, step_in_run=step_in_run,
                state=new_state, W=W, output=output,
                crashed=crash, rewarded=got_reward,
            )

        # arm_reward/turn_reward both use the same big/small-arm magnitude,
        # since a "correct" (big-reward) arm pays BIG_REWARD and a "wrong"
        # (small-reward) arm pays SMALL_REWARD at each payout point.
        chose_big_reward_arm = chosen_arm == big_reward_arm
        arm_reward = torch.where(chose_big_reward_arm,
                                 torch.full_like(total_reward, constants.BIG_REWARD),
                                 torch.full_like(total_reward, constants.SMALL_REWARD))
        turn_reward = torch.where(chose_big_reward_arm,
                                  torch.full_like(total_reward, constants.TURN_REWARD_BIG),
                                  torch.full_like(total_reward, constants.TURN_REWARD_SMALL))

        reward_delta = torch.zeros(pop, device=device)
        reward_delta = torch.where(crash, torch.full_like(reward_delta, constants.CRASH_PENALTY), reward_delta)
        reward_delta = torch.where(turned_correctly, turn_reward, reward_delta)
        reward_delta = torch.where(got_reward, arm_reward, reward_delta)
        total_reward += reward_delta
        recent_reward = reward_delta  # fed as the reward-input neuron's value on the NEXT tick

        # --- terminate / reset finished runs (only crash or reaching mazeend end a run) ---
        terminate = crash | got_reward
        if collect_tracking:
            current_turn_choice = torch.where(is_turn_tick & turn_ok, arm_choice, current_turn_choice)
            finished = torch.where(terminate)[0]
            if finished.numel() > 0:
                finished_run_idx = run_count[finished]
                tracking["decisions_by_run"][finished, finished_run_idx] = current_turn_choice[finished]
                tracking["crashed_by_run"][finished, finished_run_idx] = crash[finished]
                tracking["rewarded_by_run"][finished, finished_run_idx] = got_reward[finished]
                tracking["big_reward_by_run"][finished, finished_run_idx] = (got_reward & chose_big_reward_arm)[finished]
                tracking["correct_arm_by_run"][finished, finished_run_idx] = (
                    current_turn_choice[finished] == big_reward_arm[finished]
                )
            current_turn_choice = torch.where(terminate, torch.full_like(current_turn_choice, -1), current_turn_choice)

        run_count += terminate.long()
        step_in_run = torch.where(terminate, torch.ones_like(step_in_run), step_in_run + 1)

        new_arm = _draw_reward_arm(pop, reward_generator, device)
        big_reward_arm = torch.where(terminate, new_arm, big_reward_arm)
        new_sensory_a, new_sensory_b = _sensory_from_arm(big_reward_arm, context)
        sensory_a = torch.where(terminate, new_sensory_a, sensory_a)
        sensory_b = torch.where(terminate, new_sensory_b, sensory_b)

        if collect_tracking and finished.numel() > 0:
            next_run_idx = run_count[finished]
            valid = next_run_idx < num_runs
            if valid.any():
                vi = finished[valid]
                tracking["sensory_cue_by_run"][vi, next_run_idx[valid]] = (new_sensory_b[vi] > 0.5).long()

        state = new_state

    if collect_tracking:
        return state, W, total_reward, tracking
    return state, W, total_reward