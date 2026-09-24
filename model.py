"""Rank-one MI-GNAN: scalar shapes, categorical lookups, and exact components."""

import torch
from torch import nn


def graph_sum(values, batch, num_graphs):
    result = values.new_zeros((num_graphs,) + values.shape[1:])
    return result.index_add(0, batch, values)


def graph_mean(values, batch, num_graphs):
    """First average nodes within each graph, then average graphs equally."""
    counts = torch.bincount(batch, minlength=num_graphs).to(values.dtype)
    shape = (num_graphs,) + (1,) * (values.ndim - 1)
    return (graph_sum(values, batch, num_graphs) / counts.reshape(shape)).mean(0)


def mlp(hidden, depth, dropout, outputs):
    widths = [1] + [hidden] * (depth - 1) + [outputs]
    layers = []
    for i in range(depth):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        if i < depth - 1:
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
    # Modest starting scores avoid large sum-pooled logits.
    with torch.no_grad():
        layers[-1].weight.mul_(0.05)
        layers[-1].bias.zero_()
    return nn.Sequential(*layers)


class ShapeBank(nn.Module):
    """One function per logical input variable; output is [nodes, variables, logits]."""

    def __init__(self, cardinalities, hidden, depth, dropout, outputs):
        super().__init__()
        self.cardinalities = tuple(cardinalities)
        self.shapes = nn.ModuleList()
        for categories in cardinalities:
            if categories:
                shape = nn.Embedding(categories, outputs)
                nn.init.normal_(shape.weight, std=0.02)
            else:
                shape = mlp(hidden, depth, dropout, outputs)
            self.shapes.append(shape)

    def forward(self, x):
        return torch.stack([shape(x[:, i].long()) if self.cardinalities[i]
                            else shape(x[:, i:i + 1])
                            for i, shape in enumerate(self.shapes)], dim=1)


class MIGNAN(nn.Module):
    """F + S + A*B at each node, summed to graph logits or standardized targets.

    Each categorical group is one input variable. With multiple outputs,
    each logit has its own factors; numerical outputs share a shape's hidden
    layers. There are no interactions between different output logits.
    """

    bank_names = ("feature_main", "topology_main", "feature_factor", "topology_factor")

    def __init__(self, schema, num_motifs, hidden=16, depth=2, dropout=0., outputs=1):
        super().__init__()
        if not schema or num_motifs < 1 or outputs < 1 or hidden < 1 or depth < 1:
            raise ValueError("Nonempty features/motifs and positive dimensions are required")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self.schema = tuple(schema)
        self.config = dict(num_motifs=num_motifs, hidden=hidden, depth=depth,
                           dropout=dropout, outputs=outputs)
        cardinalities = [len(f.categories) if f.kind != "numerical" else 0 for f in schema]
        for name in self.bank_names:
            sizes = cardinalities if name.startswith("feature") else [0] * num_motifs
            setattr(self, name, ShapeBank(sizes, hidden, depth, dropout, outputs))
            self.register_buffer("mean_" + name, torch.zeros(len(sizes), outputs))
        self.intercept = nn.Parameter(torch.zeros(outputs))

    def raw_components(self, x, topology):
        return {name: getattr(self, name)(x if name.startswith("feature") else topology)
                for name in self.bank_names}

    def node_components(self, x, topology):
        parts = {name: value - getattr(self, "mean_" + name)
                 for name, value in self.raw_components(x, topology).items()}
        parts["F"] = parts["feature_main"].sum(1)
        parts["S"] = parts["topology_main"].sum(1)
        parts["I"] = parts["feature_factor"].sum(1) * parts["topology_factor"].sum(1)
        return parts

    def forward(self, batch, return_components=False):
        parts = self.node_components(batch.x, batch.topology)
        output = self.intercept + graph_sum(parts["F"] + parts["S"] + parts["I"],
                                            batch.batch, batch.num_graphs)
        return (output, parts) if return_components else output

    def sparsity(self, parts, batch):
        """Graph-balanced L1 of centered shapes; average over output logits."""
        return sum(graph_mean(parts[name].abs(), batch.batch, batch.num_graphs).sum(0).mean()
                   for name in self.bank_names)

    @torch.no_grad()
    def refresh_centering(self, batches):
        """Exact training-reference means in eval mode, fixed during each epoch."""
        device = self.intercept.device
        was_training = self.training
        self.eval()
        totals = {name: torch.zeros_like(getattr(self, "mean_" + name), dtype=torch.float64)
                  for name in self.bank_names}
        graphs = 0
        for batch in batches:
            batch = batch.to(device)
            for name, value in self.raw_components(batch.x, batch.topology).items():
                totals[name] += graph_mean(value.double(), batch.batch, batch.num_graphs) * batch.num_graphs
            graphs += batch.num_graphs
        if graphs == 0:
            raise ValueError("Centering requires training graphs")
        for name in self.bank_names:
            getattr(self, "mean_" + name).copy_(totals[name] / graphs)
        self.train(was_training)
