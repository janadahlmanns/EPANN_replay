"""HDF5 read/write for one run_evolution.py run's full numeric results: final
genomes, per-generation history, and tracked-generation snapshots -- everything the
run's plots are generated from -- plus its config and run metadata, so the run
directory alone is enough to reproduce and re-inspect the experiment later.

File layout (see save_results_h5):
    meta/                          .attrs: config_json, run_name, timestamp, ...
    meta/config/                   .attrs: same config, unwrapped into individual
                                    name/value attrs for human readability -- the
                                    packed config_json attr above is the one other
                                    scripts should parse back
    genome/final_population/       final PGPE population genome tensors + fitness
    genome/center/                 final PGPE distribution mean, as a genome
    genome/stdev/                  final PGPE distribution stdev, as a genome
    genome/best/                   best individual of the final population + fitness
    history/reward_evolution/      per-generation reward/L1 stats (see fitness.py)
    history/cue_importance/        per-tracked-generation cue-ablation importance
    history/pgpe/                  per-generation PGPE center/stdev diagnostics
    history/tracked/               per-tracked-generation snapshots, stacked along
                                    a leading "tracked generation" axis
"""

import json

import h5py
import numpy as np
import torch

from sim_core.genome_codec import unflatten_genome

# ==== 1) FILENAME ================================================================
def results_filename(run_name):
    """Results filename for one run, prefixed with its run name (config name) --
    matches the naming convention already used for that run's plot files."""
    return f"{run_name}_results.h5"


# ==== 2) GENOME SERIALIZATION ===================================================
def _write_genome_group(h5_group, genome_dict):
    """genome_dict: dict of name -> [pop, *shape] (or [1, *shape]) tensors."""
    for name, tensor in genome_dict.items():
        h5_group.create_dataset(name, data=tensor.detach().cpu().numpy())


def _write_final_genomes(h5_file, searcher):
    """Save the final PGPE population (all individuals), the distribution's
    center/stdev, and the final population's best individual -- all unflattened
    into named genome tensors (see sim_core/genome_codec.py's GENOME_SPEC)."""
    population = searcher.population
    pop_group = h5_file.create_group("genome/final_population")
    _write_genome_group(pop_group, unflatten_genome(population.values, len(population)))
    pop_group.create_dataset("fitness", data=population.evals.detach().cpu().numpy().reshape(-1))

    status = searcher.status
    center = status["center"].detach().reshape(1, -1)
    stdev = status["stdev"].detach().reshape(1, -1)
    _write_genome_group(h5_file.create_group("genome/center"), unflatten_genome(center, 1))
    _write_genome_group(h5_file.create_group("genome/stdev"), unflatten_genome(stdev, 1))

    best = status["pop_best"]
    best_group = h5_file.create_group("genome/best")
    _write_genome_group(best_group, unflatten_genome(best.values.detach().reshape(1, -1), 1))
    best_group.create_dataset("fitness", data=np.array([float(best.evals.detach().reshape(-1)[0])]))


# ==== 3) HISTORY SERIALIZATION ==================================================
def _write_flat_history(h5_group, history_dict):
    """history_dict: dict of name -> python list of scalars, one entry per generation."""
    for name, values in history_dict.items():
        h5_group.create_dataset(name, data=np.array(values))


def _write_tracked_records(h5_group, tracked_generations, tracked_records):
    """tracked_records: list of dicts (one per tracked generation), each holding
    [pop, ...]-shaped tensors. Stacked here into [num_tracked, pop, ...] arrays."""
    h5_group.create_dataset("generation", data=np.array(tracked_generations))
    if not tracked_records:
        return
    for key in tracked_records[0]:
        if key == "generation":
            continue
        stacked = torch.stack([record[key] for record in tracked_records]).numpy()
        h5_group.create_dataset(key, data=stacked)


# ==== 4) TOP-LEVEL WRITE/READ ENTRYPOINTS =======================================
def _flatten_config(config, prefix=""):
    """Recursively flatten a (possibly nested) config dict into {"a/b/c": leaf_value}.
    Blind to whatever keys config actually has, so this needs no updates when config
    keys are added/removed/renamed -- it just walks whatever dict it's given."""
    flat = {}
    for key, value in config.items():
        flat_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_config(value, prefix=f"{flat_key}/"))
        else:
            flat[flat_key] = value
    return flat


def _write_readable_config(h5_group, config):
    """Write config a second time as individual name/value attrs (on top of the
    single packed config_json attr) purely so a human browsing the .h5 in an HDF5
    viewer/h5py shell can read config values directly, without parsing JSON. Tiny
    amount of duplicated data, not meant for scripts to read back."""
    for key, value in _flatten_config(config).items():
        # h5py attrs need a plain scalar/string/array type; str() covers anything else
        # (e.g. a list of mixed types) rather than erroring on some future config shape
        h5_group.attrs[key] = value if isinstance(value, (int, float, str, bool)) else str(value)


def save_results_h5(path, config, run_metadata, searcher, history, pgpe_history):
    """Write one run's full numeric results plus its config/metadata to path."""
    with h5py.File(path, "w") as h5_file:
        meta_group = h5_file.create_group("meta")
        meta_group.attrs["config_json"] = json.dumps(config, indent=2)
        for key, value in run_metadata.items():
            meta_group.attrs[key] = str(value)  # str() guards against non-plain-str subclasses (e.g. TorchVersion)
        _write_readable_config(h5_file.create_group("meta/config"), config)

        _write_final_genomes(h5_file, searcher)

        _write_flat_history(h5_file.create_group("history/reward_evolution"), history["reward_evolution"])
        _write_flat_history(h5_file.create_group("history/cue_importance"), history["cue_importance_history"])
        _write_flat_history(h5_file.create_group("history/pgpe"), pgpe_history)
        _write_tracked_records(
            h5_file.create_group("history/tracked"),
            history["tracked_generations"],
            history["tracked_records"],
        )


def _load_group(h5_group):
    """Recursively load an h5py Group into a nested dict of numpy arrays, with a
    sibling '_attrs' dict for any attributes on that group."""
    loaded = {"_attrs": dict(h5_group.attrs)}
    for name, item in h5_group.items():
        loaded[name] = _load_group(item) if isinstance(item, h5py.Group) else item[()]
    return loaded


def load_results_h5(path):
    """Load a full results.h5 file into a nested dict mirroring the layout above."""
    with h5py.File(path, "r") as h5_file:
        return _load_group(h5_file)
