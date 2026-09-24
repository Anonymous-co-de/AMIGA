"""Exact candidate counts, reusable null statistics, and lazy motif occurrences."""

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import zipfile

import igraph as ig
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from .settings import CACHE_ROOT


@dataclass
class MotifData:
    counts: list[np.ndarray]
    occurrences: Sequence
    vocabulary: pd.DataFrame
    candidates: pd.DataFrame
    cache_path: Path | None = None
    cache_hit: bool = False
    null_counts: np.ndarray | None = None


class LazyOccurrences(Sequence):
    """Enumerate selected motifs only for graphs requested by an explanation."""

    def __init__(self, graphs, vocabulary):
        self.graphs, self.vocabulary = graphs, vocabulary
        self._items = {}

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self._items:
            _, occurrences = count_participation([self.graphs[index]], self.vocabulary)
            self._items[index] = occurrences[0]
        return self._items[index]


def as_igraph(graph):
    edges = [(int(u), int(v)) for u, v in graph.edge_index.T if u < v]
    return ig.Graph(n=graph.num_nodes, edges=edges, directed=False)


def bh_correction(p):
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    result = np.empty_like(p)
    result[order] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1)
    return result


def motif_counts(graph, plan):
    values = []
    for size, classes in plan:
        counts = graph.motifs_randesu(size=size)
        values.extend(counts[c] for c in classes)
    return np.asarray(values, dtype=np.uint64)


def null_counts(specifications, plan, rewires, seed):
    ig.set_random_number_generator(random.Random(seed))
    total = np.zeros(sum(len(classes) for _, classes in plan), dtype=np.uint64)
    try:
        for nodes, edges in specifications:
            graph = ig.Graph(n=nodes, edges=edges, directed=False)
            if graph.ecount() >= 2:
                graph.rewire(rewires * graph.ecount(), allowed_edge_types="simple")
            total += motif_counts(graph, plan)
    finally:
        ig.set_random_number_generator(None)
    return total


def mine_motifs(graphs, train_indices, sizes, null_replicates, rewires_per_edge, seed, jobs):
    plan = [(size, np.flatnonzero(np.isfinite(ig.Graph.Full(size).motifs_randesu(size=size))))
            for size in sizes]
    training = [as_igraph(graphs[i]) for i in train_indices]
    observed = np.sum([motif_counts(graph, plan) for graph in training], axis=0)
    mean = std = p = q = np.full_like(observed, np.nan, dtype=float)
    samples = np.empty((0, len(observed)), dtype=np.uint64)
    if null_replicates:
        specs = [(g.vcount(), g.get_edgelist()) for g in training]
        seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(null_replicates)]
        samples = np.stack(Parallel(n_jobs=jobs, prefer="processes")(
            delayed(null_counts)(specs, plan, rewires_per_edge, s) for s in seeds))
        mean = samples.mean(0)
        std = samples.std(0, ddof=1) if null_replicates > 1 else np.zeros_like(mean)
        p = (1 + (samples >= observed).sum(0)) / (null_replicates + 1)
        q = bh_correction(p)
    z = np.divide(observed - mean, std, out=np.full_like(mean, np.nan), where=std > 0)
    keys = [(size, int(cls)) for size, classes in plan for cls in classes]
    candidates = pd.DataFrame([
        dict(size=size, isoclass=cls, observed=int(observed[i]), null_mean=mean[i],
             null_std=std[i], z_score=z[i], p_value=p[i], q_value=q[i],
             template_edges=json.dumps(ig.Graph.Isoclass(size, cls, directed=False).get_edgelist()))
        for i, (size, cls) in enumerate(keys)
    ])
    candidates.insert(0, "candidate_column", np.arange(len(candidates)))
    return candidates, samples


def count_participation(graphs, vocabulary, store_occurrences=True):
    """Apply an already frozen vocabulary, including to previously unseen graphs."""
    if len(vocabulary) == 0 or not np.array_equal(vocabulary.column, np.arange(len(vocabulary))):
        raise ValueError("Vocabulary columns must be nonempty and numbered consecutively")
    by_size = {}
    for row in vocabulary.itertuples():
        by_size.setdefault(int(row.size), {})[int(row.isoclass)] = int(row.column)
    counts, occurrences = [], []
    for record in graphs:
        graph = as_igraph(record)
        current = {m: [] for m in range(len(vocabulary))} if store_occurrences else None
        values = np.zeros((record.num_nodes, len(vocabulary)), dtype=np.uint64)
        for size, mapping in by_size.items():
            def collect(_, vertices, isoclass):
                column = mapping.get(isoclass)
                if column is not None:
                    if store_occurrences:
                        current[column].append(tuple(sorted(vertices)))
                    values[vertices, column] += 1
                return False
            graph.motifs_randesu(size=size, callback=collect)
        counts.append(values)
        if store_occurrences:
            occurrences.append(current)
    return counts, occurrences


def cache_identity(graphs, train, config):
    """Hash structure/order, training membership, algorithm version, and settings.

    Labels/features do not affect motif counts and deliberately are not hashed.
    Selection thresholds and execution options do not affect the archive.
    """
    digest = hashlib.sha256()
    for graph in graphs:
        digest.update(np.asarray([graph.num_nodes, graph.edge_index.shape[1]], dtype="<i8").tobytes())
        digest.update(np.asarray(graph.edge_index, dtype="<i8").tobytes())
    config = {k: config[k] for k in ("sizes", "null_replicates", "rewires_per_edge", "seed", "mining_graphs")
              if k in config and config[k] is not None}
    config["sizes"] = sorted(set(config["sizes"]))
    metadata = dict(version=2, igraph=ig.__version__, graphs=digest.hexdigest(),
                    train=np.sort(train).tolist(), config=config)
    text = json.dumps(metadata, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest(), text


def compact_counts(values):
    """Choose the smallest unsigned integer dtype without losing counts."""
    maximum = int(values.max(initial=0))
    dtype = next(t for t in (np.uint8, np.uint16, np.uint32, np.uint64)
                 if maximum <= np.iinfo(t).max)
    return values.astype(dtype, copy=False)


def candidate_vocabulary(candidates):
    vocabulary = candidates.copy().reset_index(drop=True)
    vocabulary.insert(0, "motif", [f"M{i + 1}" for i in range(len(vocabulary))])
    vocabulary.insert(0, "column", np.arange(len(vocabulary)))
    return vocabulary


def save_cache(path, data, metadata):
    """Write all candidates and exact null totals as an LZMA-compressed NPZ."""
    offsets = np.cumsum([0] + [len(c) for c in data.counts])
    records = data.candidates.astype(object).where(pd.notna(data.candidates), None).to_dict("records")
    arrays = dict(metadata=np.asarray(metadata), offsets=compact_counts(offsets),
                  counts=compact_counts(np.concatenate(data.counts)),
                  null_counts=compact_counts(data.null_counts),
                  candidates=np.asarray(json.dumps(records, allow_nan=False)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_LZMA) as archive:
            for name, array in arrays.items():
                with archive.open(f"{name}.npy", "w", force_zip64=True) as handle:
                    np.lib.format.write_array(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_cache(path, metadata):
    """Read the complete candidate archive; select_topology applies a vocabulary."""
    with np.load(path, allow_pickle=False) as stored:
        if stored["metadata"].item() != metadata:
            raise ValueError("Cache metadata mismatch; use force_recompute=True")
        candidates = pd.DataFrame(json.loads(stored["candidates"].item()))
        for key in ("null_mean", "null_std", "z_score", "p_value", "q_value"):
            candidates[key] = candidates[key].astype(float)
        offsets, values = stored["offsets"], stored["counts"]
        counts = [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        samples = stored["null_counts"]
    return MotifData(counts, [], candidate_vocabulary(candidates), candidates, Path(path), True, samples)


def select_topology(data, graphs, fdr=.05, vocabulary_mode="enriched", max_motifs=None, vocabulary=None):
    """Select from cached candidates, or apply a checkpoint's exact vocabulary."""
    candidates = data.candidates.copy()
    candidates["significant"] = (candidates.observed > candidates.null_mean) & (candidates.q_value <= fdr)
    if vocabulary is None:
        selected = (candidates[candidates.significant].sort_values(["q_value", "z_score"], ascending=[True, False])
                    if vocabulary_mode == "enriched" else
                    candidates[candidates.observed > 0].sort_values("observed", ascending=False))
        vocabulary = candidate_vocabulary(selected.head(max_motifs) if max_motifs else selected)
    if vocabulary.empty:
        raise ValueError("No motifs selected; counts are cached. Adjust fdr or explicitly choose 'all'")
    lookup = {(r.size, r.isoclass): r.candidate_column for r in candidates.itertuples()}
    columns = [lookup[(r.size, r.isoclass)] for r in vocabulary.itertuples()]
    counts = [c[:, columns] for c in data.counts]
    return MotifData(counts, LazyOccurrences(graphs, vocabulary), vocabulary, candidates,
                     data.cache_path, data.cache_hit, data.null_counts)


def build_topology(graphs, train_indices, sizes=(3, 4, 5), vocabulary_mode="enriched",
                   null_replicates=500, rewires_per_edge=60, fdr=.05, max_motifs=None,
                   seed=31415, jobs=8, cache_dir=CACHE_ROOT, force_recompute=False,
                   mining_graphs=None):
    """Mine on train, count all graphs, and reuse a matching on-disk cache.

    Use vocabulary_mode='all', null_replicates=0 for a quick illustrative run.
    An empty enriched vocabulary raises an error, without changing selection rules.
    mining_graphs caps discovery to a deterministic subset of train. Counts still
    cover every graph, and preprocessing/centering still use the full train split.
    """
    if not sizes or any(not isinstance(s, (int, np.integer)) or not 3 <= s <= 6 for s in sizes):
        raise ValueError("Motif sizes must be integers between 3 and 6")
    sizes = tuple(sorted(set(int(s) for s in sizes)))
    train = np.asarray(train_indices)
    if (train.ndim != 1 or not np.issubdtype(train.dtype, np.integer) or len(train) == 0 or
            len(np.unique(train)) != len(train) or train.min() < 0 or train.max() >= len(graphs)):
        raise ValueError("Invalid training graph indices")
    train = np.sort(train)
    if vocabulary_mode not in {"enriched", "all"} or null_replicates < 0 or rewires_per_edge < 1:
        raise ValueError("Invalid motif mining settings")
    if not 0 < fdr <= 1 or (max_motifs is not None and max_motifs < 1):
        raise ValueError("Invalid FDR or vocabulary limit")
    if vocabulary_mode == "enriched" and null_replicates == 0:
        raise ValueError("Enrichment requires null replicates")
    config = dict(sizes=sizes, vocabulary_mode=vocabulary_mode, null_replicates=null_replicates,
                  rewires_per_edge=rewires_per_edge, fdr=fdr, max_motifs=max_motifs, seed=seed)
    mining_train = train
    if mining_graphs is not None:
        if not isinstance(mining_graphs, (int, np.integer)) or mining_graphs < 1:
            raise ValueError("mining_graphs must be a positive integer or None")
        config["mining_graphs"] = int(mining_graphs)
        mining_train = np.sort(np.random.default_rng(seed).choice(
            train, min(mining_graphs, len(train)), replace=False))
    key, metadata = cache_identity(graphs, train, config)
    path = Path(cache_dir) / f"candidates_{key}.npz" if cache_dir is not None else None
    if path is not None and path.exists() and not force_recompute:
        data = load_cache(path, metadata)
    else:
        candidates, samples = mine_motifs(graphs, mining_train, sizes, null_replicates, rewires_per_edge, seed, jobs)
        vocabulary = candidate_vocabulary(candidates)
        counts, _ = count_participation(graphs, vocabulary, store_occurrences=False)
        data = MotifData(counts, [], vocabulary, candidates, path, null_counts=samples)
        if path is not None:
            save_cache(path, data, metadata)
    return select_topology(data, graphs, fdr, vocabulary_mode, max_motifs)
