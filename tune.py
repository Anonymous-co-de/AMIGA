"""Tune MI-GNAN across seeds, log validation results, and retain winning models."""

# Edit the search space here. Every combination runs on all configured seeds.
SEARCH_GRID = {
    "hidden": [16, 32],
    "depth": [2],
    "dropout": [0.0,0.3],
    "lr": [0.01,0.001],
    "weight_decay": [1e-5],
    "lambda_sparse": [1e-3],
}
SEEDS = (11, 22, 33, 44, 55)

import time
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
from sklearn.model_selection import ParameterGrid
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mignan.data import Preprocessor, load_dataset, loader, split_graphs
from mignan.model import MIGNAN
from mignan.motifs import build_topology
from mignan.settings import CACHE_ROOT, DATASETS, MODEL_SETTINGS, ROOT, TRAIN_SETTINGS, MOTIF_SETTINGS
from mignan.training import evaluate, fit, save_model, seed_everything


def dataset_fingerprint(graphs):
    """Identify graph order, features, edges, and scalar targets without saving data."""
    digest = hashlib.sha256()
    for graph in graphs:
        for value in (graph.x, graph.edge_index, np.asarray([graph.y], dtype=np.float64)):
            value = np.ascontiguousarray(value)
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def experiment_metadata(dataset, graphs, *, max_graphs, subset_seed, split_seed,
                        motif_options, cache_path, model_options, training_options):
    """Small manifest shared by tuning and the training notebook."""
    source = hashlib.sha256()
    for path in sorted(ROOT.glob("*.py")):
        source.update(path.name.encode())
        source.update(path.read_bytes())
    config = DATASETS[dataset]
    return dict(
        format_version=1, dataset=dataset, task=config["task"], target=config["target"],
        max_graphs=max_graphs, subset_seed=subset_seed, split_seed=split_seed,
        graph_count=len(graphs), data_sha256=dataset_fingerprint(graphs),
        model_options=model_options,
        training_options={k: v for k, v in training_options.items() if k not in {"seed", "verbose"}},
        motif_options={k: v for k, v in motif_options.items() if k not in {"cache_dir", "force_recompute"}},
        motif_cache=Path(cache_path).name if cache_path else None,
        environment=dict(
            python=platform.python_version(), torch_threads=torch.get_num_threads(),
            cuda=torch.version.cuda, source_sha256=source.hexdigest(),
            packages={name: version(name) for name in
                      ("torch", "torch-geometric", "igraph", "numpy", "pandas", "scikit-learn", "joblib")},
        ),
    )


def run_search(dataset, output, *, grid=None, seeds=SEEDS, max_graphs=None,
               subset_seed=9182, split_seed=2027, training_overrides=None,
               motif_overrides=None, cache_dir=CACHE_ROOT, threads=2):
    """Select by mean validation score on a fixed split; test only the winners.

    Seeds vary initialization, minibatch order, and dropout. The split, motif
    vocabulary, and preprocessing are shared by all trials. Ties keep the first
    configuration in ParameterGrid order. Losing weights live only in memory.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if len(seeds) == 0 or len(set(seeds)) != len(seeds) or any(
        not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < 2**32 for s in seeds
    ):
        raise ValueError("Provide one or more distinct integer seeds in [0, 2**32)")
    if threads < 1:
        raise ValueError("threads must be positive")
    grid = SEARCH_GRID if grid is None else grid
    allowed = set(MODEL_SETTINGS) | (set(TRAIN_SETTINGS) - {"seed", "device"})
    if set(grid) - allowed:
        raise ValueError(f"Unsupported search parameters: {sorted(set(grid) - allowed)}")
    candidates = list(ParameterGrid(grid))
    config = DATASETS[dataset]
    training = {**TRAIN_SETTINGS, **config.get("training", {}), **(training_overrides or {})}
    training.pop("seed", None)
    training.pop("verbose", None)
    if training["device"] == "auto":
        training["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    motif_options = {**MOTIF_SETTINGS, **config.get("motifs", {}), **(motif_overrides or {})}
    torch.set_num_threads(threads)
    graphs = load_dataset(dataset, max_graphs=max_graphs, seed=subset_seed)
    train, validation, test = split_graphs(graphs, task=config["task"], seed=split_seed)
    motifs = build_topology(graphs, train, cache_dir=cache_dir, **motif_options)
    preprocessing = Preprocessor.fit(graphs, motifs.counts, train, config["features"], config["task"])
    records = preprocessing.transform(graphs, motifs.counts)
    binary = config["task"] == "classification" and preprocessing.outputs == 1
    metric = "auroc" if binary else "rmse" if config["task"] == "regression" else "loss"
    best_score = -np.inf if binary else np.inf
    winners = None
    search_rows = []
    print(f"{dataset}: {len(candidates)} configurations × {len(seeds)} seeds; validation {metric}", flush=True)

    for index, params in enumerate(candidates, 1):
        model_options = {**MODEL_SETTINGS, **{k: v for k, v in params.items() if k in MODEL_SETTINGS}}
        train_options = {**training, **{k: v for k, v in params.items() if k in TRAIN_SETTINGS}}
        trials = []
        configuration_rows = []
        for seed in seeds:
            seed_everything(seed)
            model = MIGNAN(preprocessing.schema, len(motifs.vocabulary),
                           outputs=preprocessing.outputs, **model_options)
            start_time = time.time()
            result = fit(model, records, train, validation, preprocessing,
                         seed=seed, verbose=False, **train_options)
            training_time = time.time() - start_time
            trials.append((model.cpu(), dict(seed=seed, checkpoint=f"seed_{seed}.pt",
                                             best_epoch=result.best_epoch,
                                             validation_score=result.best_validation,
                                             training_time=training_time)))
            # fit restores this checkpoint; keep its metrics, not the epoch history.
            best_epoch = result.history.loc[result.history.epoch == result.best_epoch].iloc[0]
            configuration_rows.append(dict(
                config_id=index, seed=seed, **model_options, **train_options,
                best_epoch=result.best_epoch, selection_metric=metric,
                validation_score=result.best_validation,
                training_time=training_time,
                **{f"validation_{key[4:]}": float(value) if np.isfinite(value) else None
                   for key, value in best_epoch.items() if key.startswith("val_")},
            ))
            print(f"[{index}/{len(candidates)}] seed={seed}: {metric}={result.best_validation:.6g}", flush=True)
            del result
        scores = [row["validation_score"] for _, row in trials]
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        for row in configuration_rows:
            row.update(validation_mean=mean, validation_std=std)
        search_rows.extend(configuration_rows)
        print(f"  mean={mean:.6g}, std={std:.6g}, params={params}", flush=True)
        improved = mean > best_score if binary else mean < best_score
        if improved:
            best_score = mean
            best_config_id = index
            winners = trials
            best_params = params.copy()
            best_model_options, best_training_options = model_options, train_options

    # Only the selected configuration reaches the test set and checkpoint storage.
    metadata = experiment_metadata(
        dataset, graphs, max_graphs=max_graphs, subset_seed=subset_seed, split_seed=split_seed,
        motif_options=motif_options, cache_path=motifs.cache_path,
        model_options=best_model_options, training_options=best_training_options,
    )
    metadata["search"] = dict(
        grid=grid, seeds=list(seeds), metric=metric, direction="maximize" if binary else "minimize",
        best_params=best_params, validation_mean=best_score,
        validation_std=float(np.std([row["validation_score"] for _, row in winners])),
        best_config_id=best_config_id, results_csv="search_results.csv",
    )
    metadata["models"] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tuning-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for model, row in winners:
            row["test_metrics"], _ = evaluate(
                model, loader(records, test, best_training_options["batch_size"]), preprocessing)
            # Undefined reporting metrics (e.g. R² for a single test graph) use JSON null.
            row["test_metrics"] = {k: v if np.isfinite(v) else None for k, v in row["test_metrics"].items()}
            save_model(staging / row["checkpoint"], model, preprocessing, motifs.vocabulary)
            metadata["models"].append(row)
        np.savez_compressed(staging / "split.npz", train=train, validation=validation, test=test)
        for row in search_rows:
            row["selected"] = row["config_id"] == best_config_id
        with (staging / "search_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(search_rows[0]))
            writer.writeheader()
            writer.writerows(search_rows)
        (staging / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(f"Output appeared during training: {output}")
        staging.rename(output)
    print(f"Saved search results and {len(seeds)} winning checkpoints to {output}", flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--output", type=Path, help="New directory; existing runs are never overwritten")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="One or more distinct seeds; defaults to SEEDS at the top of this file")
    parser.add_argument("--max-graphs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--subset-seed", type=int, default=9182)
    parser.add_argument("--split-seed", type=int, default=2027)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    args = parser.parse_args(argv)
    overrides = {k: getattr(args, k) for k in ("epochs", "patience", "device") if getattr(args, k) is not None}
    output = args.output or ROOT / "results" / "tuning" / args.dataset / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return run_search(
        args.dataset, output, seeds=args.seeds, max_graphs=args.max_graphs,
        subset_seed=args.subset_seed, split_seed=args.split_seed,
        training_overrides=overrides,
        cache_dir=args.cache_dir, threads=args.threads,
    )


if __name__ == "__main__":
    main()
