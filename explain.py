"""Exact graph, node, feature-pair, and induced-occurrence explanations."""

import json

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from .data import Batch


@torch.no_grad()
def explain_graph(model, graph, counts, occurrences, preprocessing, vocabulary, output_index=0):
    """Explain one logit, or regression in original target units.

    Each categorical variable contributes one row (the whole one-hot group).
    Occurrence allocations subtract the zero-count response before dividing by
    node participation. They describe the predictor, not effects of graph edits.
    """
    if not 0 <= output_index < model.config["outputs"]:
        raise ValueError("Invalid output logit index")
    if len(vocabulary) != counts.shape[1] or not np.array_equal(vocabulary.column, np.arange(len(vocabulary))):
        raise ValueError("Vocabulary must match the ordered motif columns")
    device = model.intercept.device
    x, topology = preprocessing.inputs(graph, counts)
    x, topology = x.to(device), topology.to(device)
    batch = Batch(x, topology, x.new_zeros(1), torch.zeros(len(x), dtype=torch.long, device=device),
                  torch.zeros(1, dtype=torch.long, device=device))
    was_training = model.training
    model.eval()
    try:
        output, parts = model(batch, return_components=True)
        zero_topology = topology.new_tensor(-preprocessing.topology_mean / preprocessing.topology_scale)
        zero_parts = model.node_components(x, zero_topology.expand_as(topology))
    finally:
        model.train(was_training)
    values = {name: parts[name][:, :, output_index].double().cpu().numpy() for name in model.bank_names}
    zeros = {name: zero_parts[name][:, :, output_index].double().cpu().numpy()
             for name in ("topology_main", "topology_factor")}
    scale = preprocessing.target_scale if preprocessing.task == "regression" else 1.
    shift = preprocessing.target_mean if preprocessing.task == "regression" else 0.
    feature = values["feature_main"] * scale
    motif = values["topology_main"] * scale
    pairs = values["feature_factor"][:, :, None] * values["topology_factor"][:, None, :] * scale
    baseline_motif = zeros["topology_main"].sum(0) * scale
    baseline_pairs = np.einsum("nd,nm->dm", values["feature_factor"], zeros["topology_factor"]) * scale
    delta_motif = (values["topology_main"] - zeros["topology_main"]) * scale
    delta_pairs = values["feature_factor"][:, :, None] * (
        values["topology_factor"] - zeros["topology_factor"])[:, None, :] * scale
    allocated_motif = np.zeros(counts.shape[1])
    allocated_pairs = np.zeros(pairs.shape[1:])
    occurrence_rows = []
    for row in vocabulary.itertuples():
        m = int(row.column)
        incidence = np.zeros(graph.num_nodes)
        for index, nodes in enumerate(occurrences.get(m, [])):
            nodes = np.asarray(nodes, dtype=int)
            if (len(nodes) != row.size or len(np.unique(nodes)) != len(nodes) or
                    np.any(nodes < 0) or np.any(nodes >= graph.num_nodes) or np.any(counts[nodes, m] <= 0)):
                raise ValueError("Invalid motif occurrence")
            incidence[nodes] += 1
            structural = (delta_motif[nodes, m] / counts[nodes, m]).sum()
            interaction = (delta_pairs[nodes, :, m] / counts[nodes, m, None]).sum(0)
            allocated_motif[m] += structural
            allocated_pairs[:, m] += interaction
            occurrence_rows.append(dict(motif=row.motif, occurrence=index, nodes=tuple(nodes.tolist()),
                                        structural=float(structural), interaction=float(interaction.sum()),
                                        pairs=interaction))
        if not np.array_equal(incidence, counts[:, m]):
            raise ValueError("Occurrences do not reconstruct node participation counts")
    intercept = float(model.intercept[output_index].detach().cpu()) * scale + shift
    direct = float(output[0, output_index].cpu()) * scale + shift
    reconstructed = intercept + feature.sum() + motif.sum() + pairs.sum()
    errors = dict(graph=abs(direct - reconstructed),
                  motif_occurrences=float(np.max(np.abs(baseline_motif + allocated_motif - motif.sum(0)))),
                  pair_occurrences=float(np.max(np.abs(baseline_pairs + allocated_pairs - pairs.sum(0)))))
    tolerance = 1e-5 * max(1., abs(intercept) + np.abs(feature).sum() + np.abs(motif).sum() + np.abs(pairs).sum())
    if max(errors.values()) > tolerance:
        raise AssertionError(f"Incomplete decomposition: {errors}")
    names = [f.name for f in preprocessing.schema]
    motif_names = vocabulary.motif.tolist()
    if preprocessing.task == "regression":
        prediction = direct
    elif preprocessing.outputs == 1:
        prediction = float(output[0, 0].sigmoid().cpu())
    else:
        prediction = output[0].softmax(0).cpu().numpy()
    return dict(output=direct, prediction=prediction, intercept=intercept, output_index=output_index,
                units="target" if preprocessing.task == "regression" else "logit",
                channels=pd.Series(dict(intercept=intercept, F=feature.sum(), S=motif.sum(), I=pairs.sum())),
                nodes=pd.DataFrame(dict(F=feature.sum(1), S=motif.sum(1), I=pairs.sum((1, 2)))),
                features=pd.Series(feature.sum(0), index=names),
                motifs=pd.Series(motif.sum(0), index=motif_names),
                pairs=pd.DataFrame(pairs.sum(0), index=names, columns=motif_names),
                zero_motifs=pd.Series(baseline_motif, index=motif_names),
                zero_pairs=pd.DataFrame(baseline_pairs, index=names, columns=motif_names),
                occurrences=pd.DataFrame(occurrence_rows, columns=["motif", "occurrence", "nodes", "structural", "interaction", "pairs"]),
                feature_by_node=feature, motif_by_node=motif, pair_by_node=pairs, errors=errors)


def plot_explanation(explanation, graph):
    """Signed channel bars, pair matrix, and node contributions on a common scale."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    values = explanation["channels"]
    axes[0, 0].bar(values.index, values, color=["#bf4058" if v >= 0 else "#326ba5" for v in values])
    axes[0, 0].axhline(0, color="gray", lw=.7)
    axes[0, 0].set(title=f"Output = {explanation['output']:.3f}", ylabel=explanation["units"])
    pairs = explanation["pairs"]
    limit = max(1e-8, np.abs(pairs.to_numpy()).max())
    im = axes[0, 1].imshow(pairs, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[0, 1].set(xticks=range(len(pairs.columns)), xticklabels=pairs.columns,
                   yticks=range(len(pairs.index)), yticklabels=pairs.index, title="Variable × motif")
    fig.colorbar(im, ax=axes[0, 1], shrink=.7)
    network = nx.Graph()
    network.add_nodes_from(range(graph.num_nodes))
    network.add_edges_from(graph.edge_index.T)
    positions = nx.spring_layout(network, seed=17)
    nodes = explanation["nodes"]
    views = {"F + S + I": nodes.sum(axis=1), **{name: nodes[name] for name in nodes}}
    limit = max(1e-8, max(np.abs(v).max() for v in views.values()))
    for ax, (name, values) in zip([axes[0, 2], *axes[1]], views.items()):
        nx.draw_networkx(network, positions, ax=ax, node_color=values, cmap=plt.get_cmap("RdBu_r"),
                         vmin=-limit, vmax=limit, node_size=220, font_size=7, edge_color="#aaa")
        ax.set_title(f"{name}: {values.sum():+.3f}")
        ax.set_axis_off()
    return fig


def plot_occurrence(graph, vocabulary, allocation):
    """Highlight one occurrence and display its allocated structural/interaction score."""
    network = nx.Graph()
    network.add_nodes_from(range(graph.num_nodes))
    network.add_edges_from(graph.edge_index.T)
    selected = set(allocation["nodes"])
    positions = nx.spring_layout(network, seed=17)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    row = vocabulary[vocabulary.motif == allocation["motif"]].iloc[0]
    template = nx.Graph()
    template.add_nodes_from(range(int(row["size"])))
    template.add_edges_from(json.loads(row.template_edges))
    nx.draw_networkx(template, ax=axes[0], node_color="#b6caa3", node_size=300)
    axes[0].set_title(str(row.motif))
    nx.draw_networkx(network, positions, ax=axes[1], font_size=7, node_size=220,
                     node_color=["#b6caa3" if n in selected else "#ddd" for n in network])
    nx.draw_networkx_edges(network, positions, ax=axes[1], width=3, edge_color="#557a39",
                           edgelist=[(u, v) for u, v in network.edges if u in selected and v in selected])
    axes[1].set_title(f"Occurrence S={allocation['structural']:+.3f}, I={allocation['interaction']:+.3f}")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    return fig
