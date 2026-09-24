"""Graph records, explicit feature encoding, train-only scaling, and batching."""

from dataclasses import dataclass
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .settings import DATASETS, Feature


@dataclass
class Graph:
    x: np.ndarray
    edge_index: np.ndarray
    y: float = 0.0

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float32)
        if self.x.ndim != 2 or len(self.x) == 0 or not np.isfinite(self.x).all():
            raise ValueError("x must be a finite, nonempty [nodes, raw_features] matrix")
        edges = np.asarray(self.edge_index)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if not np.isfinite(edges).all() or not np.equal(edges, np.floor(edges)).all():
            raise ValueError("Edge endpoints must be integer node IDs")
        if edges.size and (edges.min() < 0 or edges.max() >= len(self.x)):
            raise ValueError("Edge endpoint outside graph")
        pairs = {(int(u), int(v)) for u, v in edges.T if u != v}
        pairs |= {(v, u) for u, v in pairs}
        self.edge_index = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2).T
        if not np.isfinite(self.y):
            raise ValueError("Graph target must be finite")

    @property
    def num_nodes(self):
        return len(self.x)


def load_dataset(name="mutagenicity", max_graphs=None, seed=9182):
    """Load TU, local adjacency/feature/target pickles, or one QM9 target.

    Subsample before constructing Graph records, retaining source order.
    All features are checked against the schema declared in settings.py.
    """
    from torch_geometric.datasets import QM9, TUDataset

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {tuple(DATASETS)}")
    if max_graphs is not None and (not isinstance(max_graphs, (int, np.integer)) or max_graphs < 1):
        raise ValueError("max_graphs must be a positive integer or None")
    config = DATASETS[name]
    kind = config["loader"]
    if kind == "tu":
        source = TUDataset(str(config["path"]), config["name"], use_node_attr=True)
        size = len(source)
    elif kind == "qm9":
        source = QM9(str(config["path"]))
        size = len(source)
    elif kind == "pickle":
        with config["path"].open("rb") as handle:
            adjacency, features, targets = pickle.load(handle)[:3]
        size = len(targets)
        if not len(adjacency) == len(features) == size:
            raise ValueError("Expected one adjacency and feature matrix per target")
    else:
        raise ValueError(f"Unknown loader: {kind}")
    if size == 0:
        raise ValueError("Dataset is empty")
    indices = np.arange(size)
    if max_graphs is not None:
        indices = np.sort(np.random.default_rng(seed).choice(indices, min(max_graphs, size), replace=False))
    graphs = []
    for i in indices:
        if kind == "pickle":
            x = np.asarray(features[i], dtype=np.float32)
            matrix = np.asarray(adjacency[i])
            if matrix.shape != (len(x), len(x)):
                raise ValueError(f"Graph {i}: adjacency must match the node count")
            edges = np.vstack(np.nonzero(matrix))
            target = np.asarray(targets[i]).reshape(-1)
        else:
            item = source[int(i)]
            if item.x is None:
                raise ValueError(f"Graph {i}: missing node features")
            x = item.x.detach().cpu().numpy()
            edges = item.edge_index.detach().cpu().numpy()
            target = item.y.detach().cpu().numpy().reshape(-1)
        if kind == "qm9":
            y = float(target[config["target_index"]])
        else:
            if target.size != 1:
                raise ValueError(f"Graph {i}: expected a scalar target")
            y = float(target[0])
        graph = Graph(x, edges, y)
        encode_features(graph.x, config["features"])
        graphs.append(graph)
    return graphs


def split_graphs(graphs, task="classification", seed=2027):
    """Deterministic 70/15/15 split; classification is stratified by label."""
    if task not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")
    indices = np.arange(len(graphs))
    strata = np.asarray([g.y for g in graphs]) if task == "classification" else None
    train, rest = train_test_split(indices, test_size=0.30, stratify=strata, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, random_state=seed + 1,
                                 stratify=None if strata is None else strata[rest])
    return tuple(np.sort(part) for part in (train, val, test))


def encode_features(x, schema):
    """Collapse one-hot groups and validate declared category domains."""
    if not schema or len({f.name for f in schema}) != len(schema):
        raise ValueError("Declare at least one feature, with unique names")
    columns = [column for f in schema for column in f.columns]
    if len(set(columns)) != len(columns) or sorted(columns) != list(range(x.shape[1])):
        raise ValueError("The schema must cover every raw column exactly once")
    values = []
    for feature in schema:
        raw = x[:, feature.columns]
        if feature.kind == "one_hot":
            if not (np.isin(raw, [0, 1]).all() and np.all(raw.sum(axis=1) == 1)):
                raise ValueError(f"{feature.name}: expected exactly one active category per node")
            value = raw.argmax(axis=1)
        else:
            value = raw[:, 0]
        if feature.kind != "numerical":
            if not (np.equal(value, np.floor(value)).all() and
                    np.all((value >= 0) & (value < len(feature.categories)))):
                raise ValueError(f"{feature.name}: invalid category ID")
        values.append(value)
    return np.column_stack(values).astype(np.float32)


def moments(values):
    # Float64 accumulation keeps constant float32 columns constant even when
    # many nodes are pooled (e.g. the 0.1-valued features in BAReg2).
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    return mean, np.where(scale > 1e-8, scale, 1.0)


@dataclass
class Preprocessor:
    schema: tuple[Feature, ...]
    task: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    topology_mean: np.ndarray
    topology_scale: np.ndarray
    target_mean: float
    target_scale: float
    classes: np.ndarray

    @classmethod
    def fit(cls, graphs, counts, train_indices, schema, task):
        """Fit only on training graphs. Every node has equal input-scaling weight."""
        train = np.asarray(train_indices, dtype=int)
        if task not in {"classification", "regression"} or len(train) == 0:
            raise ValueError("Specify a valid task and nonempty training indices")
        x = np.concatenate([encode_features(graphs[i].x, schema) for i in train])
        mean, scale = moments(x)
        categorical = np.array([f.kind != "numerical" for f in schema])
        mean[categorical], scale[categorical] = 0, 1
        top_mean, top_scale = moments(np.concatenate([
            np.log1p(np.asarray(counts[i], dtype=np.float64)) for i in train]))
        y = np.asarray([graphs[i].y for i in train], dtype=np.float64)
        classes = np.unique(y) if task == "classification" else np.array([])
        if task == "classification" and len(classes) < 2:
            raise ValueError("Training requires at least two classes")
        target_mean, target_scale = moments(y) if task == "regression" else (0., 1.)
        return cls(tuple(schema), task, mean, scale, top_mean, top_scale,
                   float(target_mean), float(target_scale), classes)

    @property
    def outputs(self):
        return len(self.classes) if len(self.classes) > 2 else 1

    def inputs(self, graph, counts):
        """Transform one graph, also for inference on graphs without known targets."""
        counts = np.asarray(counts)
        if (counts.shape != (graph.num_nodes, len(self.topology_mean)) or
                not np.isfinite(counts).all() or np.any(counts < 0)):
            raise ValueError("Invalid motif participation matrix")
        x = (encode_features(graph.x, self.schema) - self.feature_mean) / self.feature_scale
        topology = (np.log1p(np.asarray(counts, dtype=np.float64)) - self.topology_mean) / self.topology_scale
        return torch.tensor(x, dtype=torch.float32), torch.tensor(topology, dtype=torch.float32)

    def transform(self, graphs, counts):
        if len(graphs) != len(counts):
            raise ValueError("One motif matrix is required per graph")
        records = []
        for index, (graph, count) in enumerate(zip(graphs, counts)):
            x, topology = self.inputs(graph, count)
            if self.task == "classification":
                match = np.flatnonzero(self.classes == graph.y)
                if len(match) != 1:
                    raise ValueError(f"Target {graph.y} is absent from training classes")
                y = int(match[0])
            else:
                y = (graph.y - self.target_mean) / self.target_scale
            records.append((x, topology, float(y), index))
        return records


@dataclass
class Batch:
    x: torch.Tensor
    topology: torch.Tensor
    y: torch.Tensor
    batch: torch.Tensor
    graph_ids: torch.Tensor

    @property
    def num_graphs(self):
        return len(self.graph_ids)

    def to(self, device):
        return Batch(*(getattr(self, name).to(device) for name in self.__dataclass_fields__))


def collate(records):
    x, topology, y, ids = zip(*records)
    return Batch(torch.cat(x), torch.cat(topology), torch.tensor(y, dtype=torch.float32),
                 torch.repeat_interleave(torch.arange(len(x)), torch.tensor([len(v) for v in x])),
                 torch.tensor(ids))


def loader(records, indices, batch_size=128, shuffle=False):
    return DataLoader([records[int(i)] for i in indices], batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate)
