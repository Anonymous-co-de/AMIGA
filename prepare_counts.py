"""Prepare and verify compressed candidate archives for the configured datasets."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mignan.data import load_dataset, split_graphs
from mignan.motifs import build_topology, cache_identity, load_cache
from mignan.settings import CACHE_ROOT, DATASETS, MOTIF_SETTINGS


def prepare(datasets, output=CACHE_ROOT, split_seed=2027, jobs=None):
    """Use full datasets and configured mining limits; resume matching archives."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    shared = {}
    for name in datasets:
        start = time.monotonic()
        config = DATASETS[name]
        options = {**MOTIF_SETTINGS, **config.get("motifs", {})}
        if jobs is not None:
            options["jobs"] = jobs
        # QM9 targets share graph order, topology settings, and an unstratified split.
        group = (str(config["path"]), json.dumps(options, sort_keys=True))
        if config["loader"] == "qm9" and group in shared:
            index[name] = dict(shared[group])
        else:
            print(f"{name}: loading full dataset", flush=True)
            graphs = load_dataset(name)
            train, _, _ = split_graphs(graphs, task=config["task"], seed=split_seed)
            limit = options.get("mining_graphs") or len(train)
            print(f"{name}: mining on {min(len(train), limit)} "
                  f"training graphs; counting all {len(graphs)} graphs", flush=True)
            # Selection does not affect the archive; 'all' also handles empty enrichment.
            data = build_topology(graphs, train, **{**options, "vocabulary_mode": "all"}, cache_dir=output)
            path = data.cache_path
            del data
            _, metadata = cache_identity(graphs, train, options)
            raw = load_cache(path, metadata)
            mining_train = train
            if options.get("mining_graphs") is not None:
                mining_train = np.sort(np.random.default_rng(options["seed"]).choice(
                    train, min(options["mining_graphs"], len(train)), replace=False))
            # Every occurrence contributes once to each of its motif's vertices.
            incidence = np.sum([raw.counts[i].sum(0, dtype=np.uint64) for i in mining_train], axis=0)
            expected = raw.candidates.observed.to_numpy(dtype=np.uint64) * raw.candidates["size"].to_numpy(dtype=np.uint64)
            if not np.array_equal(incidence, expected):
                raise ValueError(f"{name}: candidate counts disagree with mining totals")
            entry = dict(archive=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         bytes=path.stat().st_size, graphs=len(graphs), nodes=sum(g.num_nodes for g in graphs),
                         train_graphs=len(train), mining_graphs=len(mining_train), split_seed=split_seed,
                         candidates=len(raw.candidates), counts_dtype=str(raw.counts[0].dtype),
                         null_replicates=len(raw.null_counts),
                         metadata={k: v for k, v in json.loads(metadata).items() if k != "train"})
            index[name] = entry
            if config["loader"] == "qm9":
                shared[group] = entry
            del graphs, raw
        temporary = index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(index, indent=2) + "\n")
        temporary.replace(index_path)
        print(f"{name}: verified {index[name]['bytes'] / 2**20:.2f} MiB "
              f"({time.monotonic() - start:.1f}s)", flush=True)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="Dataset keys; defaults to all settings.py entries")
    parser.add_argument("--output", type=Path, default=CACHE_ROOT)
    parser.add_argument("--split-seed", type=int, default=2027)
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()
    names = args.datasets or list(DATASETS)
    unknown = set(names) - DATASETS.keys()
    if unknown:
        parser.error(f"Unknown datasets: {sorted(unknown)}")
    prepare(names, args.output, args.split_seed, args.jobs)


if __name__ == "__main__":
    main()
