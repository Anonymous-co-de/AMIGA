"""Data and plotting helpers for the single-seed MI-GNAN toy notebook."""

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
import json

import igraph as ig
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch

from .data import Batch, Graph, split_graphs
from .motifs import count_participation
from .settings import Feature

TRUE_COLOR = "#237a75"
MODEL_COLOR = "#e29545"
COMPONENT_NAMES = ["f1", "f2", "s1", "s2", "a1*b1", "a1*b2", "a2*b1", "a2*b2"]
ZERO_COMPONENTS = ["f2", "s2", "a1*b1", "a1*b2", "a2*b1"]


@dataclass
class ToyEffects:
    alpha: float = 3.0
    beta: float = 3.0
    gamma: float = 3.0
    path_scale: float = 5.0
    triangle_scale: float = 2.0

    def feature_effect(self, x):
        return self.alpha * np.sin(np.asarray(x, dtype=np.float64))

    def path_effect(self, m):
        return self.beta * -np.expm1(-np.asarray(m, dtype=np.float64) / self.path_scale)

    def interaction_effect(self, x, m):
        return (self.gamma * np.tanh(np.asarray(x, dtype=np.float64))
                * -np.expm1(-np.asarray(m, dtype=np.float64) / self.triangle_scale))

    def node_terms(self, graph, counts):
        return np.column_stack([
            self.feature_effect(graph.x[:, 0]),
            self.path_effect(counts[:, 0]),
            self.interaction_effect(graph.x[:, 1], counts[:, 1]),
        ])


def channels_from_components(components):
    return pd.DataFrame({
        "F": components[["f1", "f2"]].sum(axis=1),
        "S": components[["s1", "s2"]].sum(axis=1),
        "I": components[COMPONENT_NAMES[4:]].sum(axis=1),
    })


def make_toy_data(*, n_graphs=2000, n_nodes=12, feature_range=(-2.0, 2.0),
                  edge_prob_range=(0.10, 0.40), seed=2027, split_seed=2027,
                  noise_ratio=0.10, noise_seed=8193, effects=None):
    """Generate clean effects, then add graph noise scaled on training targets."""
    effects = ToyEffects() if effects is None else effects
    structure_seed, feature_seed = np.random.SeedSequence(seed).spawn(2)
    structure_rng = np.random.default_rng(structure_seed)
    feature_rng = np.random.default_rng(feature_seed)
    u, v = np.triu_indices(n_nodes, k=1)
    graphs = []
    for _ in range(n_graphs):
        probability = structure_rng.uniform(*edge_prob_range)
        selected = structure_rng.random(len(u)) < probability
        x = feature_rng.uniform(*feature_range, size=(n_nodes, 2)).astype(np.float32)
        graphs.append(Graph(x=x, edge_index=np.vstack([u[selected], v[selected]])))

    templates = [
        ("M1", "path_length_2", [(0, 1), (1, 2)]),
        ("M2", "triangle", [(0, 1), (1, 2), (0, 2)]),
    ]
    vocabulary = pd.DataFrame([
        dict(column=column, motif=motif, name=name, size=3,
             isoclass=ig.Graph(n=3, edges=edges, directed=False).isoclass(),
             template_edges=json.dumps(edges))
        for column, (motif, name, edges) in enumerate(templates)
    ])
    counts, occurrences = count_participation(graphs, vocabulary)
    terms = [effects.node_terms(g, c) for g, c in zip(graphs, counts)]
    y_clean = np.array([values.sum() for values in terms])
    train_idx, val_idx, test_idx = split_graphs(graphs, task="regression", seed=split_seed)
    sigma = noise_ratio * y_clean[train_idx].std()
    noise = np.random.default_rng(noise_seed).normal(0.0, sigma, n_graphs)
    targets = y_clean + noise
    for graph, target in zip(graphs, targets):
        graph.y = float(target)

    true_components = []
    for graph, values in zip(graphs, terms):
        components = pd.DataFrame(0.0, index=np.arange(graph.num_nodes), columns=COMPONENT_NAMES)
        components[["f1", "s1", "a2*b2"]] = values
        true_components.append(components)
    return SimpleNamespace(
        graphs=graphs, counts=counts, occurrences=occurrences, vocabulary=vocabulary,
        y_clean=y_clean, targets=targets, sigma=sigma, effects=effects,
        feature_range=feature_range, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        schema=(Feature("f1", "numerical", (0,)), Feature("f2", "numerical", (1,))),
        true_components=true_components,
        true_nodes=[pd.DataFrame(t, columns=["F", "S", "I"]) for t in terms],
    )


@torch.no_grad()
def explain_at_zero(model, graph, participation, preprocessing):
    """Decompose the prediction at raw-zero inputs, without using the target."""
    device = model.intercept.device
    x, topology = (value.to(device) for value in preprocessing.inputs(graph, participation))
    zero_graph = Graph(
        x=np.zeros((1, graph.x.shape[1]), dtype=np.float32),
        edge_index=np.empty((2, 0), dtype=np.int64),
    )
    x0, topology0 = (value.to(device) for value in preprocessing.inputs(
        zero_graph, np.zeros((1, 2), dtype=np.uint64)
    ))
    batch = Batch(x, topology, x.new_zeros(1),
                  torch.zeros(graph.num_nodes, dtype=torch.long, device=device),
                  torch.zeros(1, dtype=torch.long, device=device))
    was_training = model.training
    model.eval()
    try:
        output, parts = model(batch, return_components=True)
        at_zero = model.node_components(x0, topology0)
    finally:
        model.train(was_training)

    values = {key: parts[key][:, :, 0].double().cpu().numpy() for key in model.bank_names}
    zeros = {key: at_zero[key][0, :, 0].double().cpu().numpy() for key in model.bank_names}
    f, s = values["feature_main"], values["topology_main"]
    a, b = values["feature_factor"], values["topology_factor"]
    f0, s0 = zeros["feature_main"], zeros["topology_main"]
    a0, b0 = zeros["feature_factor"], zeros["topology_factor"]
    da, db = a - a0, b - b0
    scale, shift = preprocessing.target_scale, preprocessing.target_mean

    feature = ((f - f0) + da * b0.sum()) * scale
    structural = ((s - s0) + db * a0.sum()) * scale
    pairs = da[:, :, None] * db[:, None, :] * scale
    components = pd.DataFrame(
        np.column_stack([feature, structural, pairs.reshape(graph.num_nodes, -1)]),
        columns=COMPONENT_NAMES,
    )
    nodes = channels_from_components(components)
    constant_per_node = float(f0.sum() + s0.sum() + a0.sum() * b0.sum()) * scale
    intercept = float(model.intercept[0].detach().cpu()) * scale + shift + graph.num_nodes * constant_per_node
    prediction = float(output[0, 0].cpu()) * scale + shift
    channels = pd.Series({"intercept": intercept, **nodes.sum(axis=0).to_dict(), "output": prediction})
    return dict(components=components, nodes=nodes, channels=channels,
                intercept=intercept, node_reference=constant_per_node, output=prediction)


def component_values(model, preprocessing, *, x1=0.0, x2=0.0, m1=0, m2=0):
    """Query component functions at the raw-zero explanation reference."""
    x1, x2, m1, m2 = np.broadcast_arrays(
        *[np.atleast_1d(value) for value in (x1, x2, m1, m2)]
    )
    query = Graph(
        x=np.column_stack([x1, x2]).astype(np.float32),
        edge_index=np.empty((2, 0), dtype=np.int64),
    )
    participation = np.column_stack([m1, m2]).astype(np.float64)
    return explain_at_zero(model, query, participation, preprocessing)["components"]


def prepare_plot_data(model, preprocessing, dataset, *, graph_position=None):
    """Compute test explanations once for the main figure and optional diagnostics."""
    ids = dataset.test_idx
    explanations = [
        explain_at_zero(model, dataset.graphs[int(i)], dataset.counts[int(i)], preprocessing)
        for i in ids
    ]
    true_graphs = [dataset.true_nodes[int(i)].to_numpy() for i in ids]
    pred_graphs = [e["nodes"].to_numpy() for e in explanations]
    if graph_position is None:
        graph_position = next(
            (j for j, i in enumerate(ids) if (dataset.counts[int(i)].sum(0) > 0).all()), 0
        )
    gid = int(ids[graph_position])
    return dict(
        graph_id=gid, graph=dataset.graphs[gid], explanation=explanations[graph_position],
        true_local=true_graphs[graph_position], pred_local=pred_graphs[graph_position],
        truth=np.concatenate(true_graphs), learned=np.concatenate(pred_graphs),
        channel_rmse=np.sqrt(np.mean([
            np.mean((p - t)**2, axis=0) for t, p in zip(true_graphs, pred_graphs)
        ], axis=0)),
        explanations=explanations, true_graphs=true_graphs, pred_graphs=pred_graphs,
        training_counts=np.concatenate([dataset.counts[int(i)] for i in dataset.train_idx]),
        effects=dataset.effects, feature_range=dataset.feature_range,
        component_values=partial(component_values, model, preprocessing),
    )


def style_axis(ax, *, title=None, xlabel=None, ylabel="Contribution (target units)"):
    ax.axhline(0, color="#59636e", linewidth=0.8)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def paired_bars(ax, truth, learned, labels, title, annotate=False):
    positions = np.arange(len(labels))
    width = 0.36
    true_bars = ax.bar(positions - width / 2, truth, width, label="True", color=TRUE_COLOR)
    model_bars = ax.bar(positions + width / 2, learned, width, label="MI-GNAN", color=MODEL_COLOR)
    ax.set_xticks(positions, labels)
    style_axis(ax, title=title)
    if annotate:
        for bars in (true_bars, model_bars):
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.margins(y=0.18)


def share_effect_limits(axes, amplitudes=(3.0, 3.0, 3.0)):
    """One symmetric y range for true/learned curves, including all deviations."""
    extent = max(*(abs(value) for value in amplitudes), 1e-8)
    for ax in axes:
        for line in ax.lines:
            values = np.asarray(line.get_ydata(), dtype=float)
            extent = max(extent, float(np.abs(values).max(initial=0)))
    limits = (-1.08 * extent, 1.08 * extent)
    for ax in axes:
        ax.set_ylim(limits)
    return limits


def plot_summary_figure(data, interaction_view="3d", *, fontsize=14, shape_linewidth=3.0):
    """Render the same test diagnostics with surfaces, heatmaps, or slices."""
    effects = data["effects"]
    ALPHA, BETA, GAMMA = effects.alpha, effects.beta, effects.gamma
    FEATURE_RANGE = data["feature_range"]
    feature_effect, path_effect = effects.feature_effect, effects.path_effect
    interaction_effect = effects.interaction_effect
    component_values = data["component_values"]
    training_counts = data["training_counts"]
    true_intercept = 0.0
    tick_fontsize = fontsize - 2
    legend_fontsize = fontsize - 2
    gid = data["graph_id"]
    g = data["graph"]
    local = data["explanation"]
    true_local, pred_local = data["true_local"], data["pred_local"]
    truth, learned = data["truth"], data["learned"]
    channel_rmse = data["channel_rmse"]

    if interaction_view not in {"3d", "heatmap", "curves"}:
        raise ValueError("Choose 3d, heatmap, or curves")
    fig = plt.figure(figsize=(12.4, 4.8), layout="constrained")
    rows = fig.add_gridspec(2, 1)
    top = rows[0].subgridspec(1, 3)
    widths = [1, 1, 1, 1, 2] if interaction_view == "curves" else [1, 1, 1, 1, 1.5, 1.5, 0.06]
    bottom = rows[1].subgridspec(1, len(widths), width_ratios=widths)
    ax_test, ax_graph, ax_nodes = [fig.add_subplot(top[j]) for j in range(3)]
    shape_axes = [fig.add_subplot(bottom[j]) for j in range(4)]
    if interaction_view == "curves":
        interaction_axes = [fig.add_subplot(bottom[4])]
    else:
        projection = "3d" if interaction_view == "3d" else None
        interaction_axes = [fig.add_subplot(bottom[j], projection=projection) for j in (4, 5)]

    flat_axes = [ax_test, ax_graph, ax_nodes, *shape_axes]
    if interaction_view != "3d":
        flat_axes += interaction_axes
    for ax in flat_axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=tick_fontsize)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)

    colors = ["#237a75", "#b57924", "#7851a9"]

    low = min(truth.min(), learned.min())
    high = max(truth.max(), learned.max())
    margin = 0.05 * max(high - low, 0.1)
    limits = (low - margin, high + margin)

    ax_test.plot(limits, limits, color="#777777", lw=1, zorder=0)
    ax_test.set(
        xlim=limits, ylim=limits,
        xlabel="True contribution", ylabel="Learned\ncontribution",
    )
    ax_test.set_title("(a) Node-level channel recovery", fontsize=fontsize)

    styles = {
        0: dict(
            marker="o", s=7, color=colors[0],
            alpha=0.30, linewidths=0,
        ),
        2: dict(
            marker="x", s=9, color=colors[2],
            alpha=0.45, linewidths=0.6,
        ),
        1: dict(
            marker="D", s=25, facecolors="none",
            edgecolors=colors[1], alpha=0.85, linewidths=0.8,
        ),
    }

    # Draw S last, with unfilled diamonds.
    for zorder, k in enumerate((0, 2, 1), start=1):
        ax_test.scatter(
            truth[:, k], learned[:, k],
            zorder=zorder, rasterized=True, **styles[k],
        )

    ax_test.legend(
        handles=[
            Line2D(
                [], [], linestyle="none",
                marker=marker, markersize=5,
                color=colors[k],
                markerfacecolor="none" if k == 1 else colors[k],
                label=f"{name}: RMSE {channel_rmse[k]:.4f}",
            )
            for k, name, marker in [
                (0, "F", "o"), (1, "S", "D"), (2, "I", "x")
            ]
        ],
        loc="upper left", frameon=False, fontsize=legend_fontsize,
        handletextpad=0.3, labelspacing=0.25,
    )

    # Graph decomposition.
    positions = np.arange(3)
    width = 0.36

    ax_graph.bar(
        positions - width / 2, true_local.sum(0),
        width, color=TRUE_COLOR, label="True",
    )
    ax_graph.bar(
        positions + width / 2, pred_local.sum(0),
        width, color=MODEL_COLOR, label="MI-GNAN",
    )
    ax_graph.axhline(0, color="gray", lw=0.6)
    ax_graph.set_xticks(positions, ["F", "S", "I"])
    ax_graph.set_ylabel("Graph\ncontribution")
    ax_graph.set_title(
        "(b) Graph-level decomposition",
        fontsize=fontsize,
    )
    ax_graph.set_xlabel(
        f"Intercept: {true_intercept:.3f} / {local['intercept']:.3f}",
        fontsize=legend_fontsize,
    )
    ax_graph.legend(frameon=False, fontsize=legend_fontsize, labelspacing=0.25)

    # Node contributions.
    node_ids = np.arange(g.num_nodes)
    true_nodes = true_local.sum(1)
    pred_nodes = pred_local.sum(1)

    ax_nodes.bar(
        node_ids - width / 2, true_nodes, width, color=TRUE_COLOR,
    )
    ax_nodes.bar(
        node_ids + width / 2, pred_nodes, width, color=MODEL_COLOR,
    )
    ax_nodes.axhline(0, color="gray", lw=0.6)
    ax_nodes.set_xticks(node_ids)
    ax_nodes.set(xlabel="Node", ylabel="F + S + I")
    ax_nodes.set_title(
        "(c) Node-level total contributions",
        fontsize=fontsize,
    )

    # Main-effect curves.
    grid = np.linspace(*FEATURE_RANGE, 250, dtype=np.float32)
    path_counts = np.unique(training_counts[:, 0]).astype(int)
    triangle_counts = np.unique(training_counts[:, 1]).astype(int)

    curves = [
        (
            grid,
            feature_effect(grid),
            component_values(x1=grid)["f1"].to_numpy(),
            r"(d) $f_1(x_1)$", r"$x_1$",
        ),
        (
            grid,
            np.zeros_like(grid),
            component_values(x2=grid)["f2"].to_numpy(),
            r"(e) $f_2(x_2)$", r"$x_2$",
        ),
        (
            path_counts,
            path_effect(path_counts),
            component_values(m1=path_counts)["s1"].to_numpy(),
            r"(f) $s_1(h_1)$", "Path count",
        ),
        (
            triangle_counts,
            np.zeros_like(triangle_counts, dtype=float),
            component_values(m2=triangle_counts)["s2"].to_numpy(),
            r"(g) $s_2(h_2)$", "Triangle count",
        ),
    ]

    for ax, (x, true_y, pred_y, title, xlabel) in zip(shape_axes, curves):
        ax.plot(x, true_y, color=TRUE_COLOR, lw=shape_linewidth)
        ax.plot(x, pred_y, color=MODEL_COLOR, lw=shape_linewidth, ls="--")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=fontsize)

    shape_axes[0].set_ylabel("Contribution at\nzero reference")
    for ax in shape_axes[2:]:
        ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))

    # Shared vertical limits for main effects.
    share_effect_limits(shape_axes, (ALPHA, BETA, GAMMA))

    if interaction_view == "curves":
        ax_interaction = interaction_axes[0]
        levels = np.unique(np.quantile(
            training_counts[:, 1], [0, 0.5, 0.9], method="nearest"
        )).astype(int)

        interaction_colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(levels)))

        for count, color in zip(levels, interaction_colors):
            true_y = (
                interaction_effect(grid, count)
            )
            pred_y = component_values(x2=grid, m2=count)["a2*b2"].to_numpy()

            ax_interaction.plot(
                grid, true_y, color=color, lw=1.7,
                label=fr"$h_2={count}$",
            )
            ax_interaction.plot(
                grid, pred_y, color=color, lw=1.7, ls="--",
            )

        ax_interaction.axhline(0, color="gray", lw=0.5)
        ax_interaction.set_xlabel(r"$x_2$")
        ax_interaction.set_title(r"(h) $a_2(x_2)b_2(h_2)$", fontsize=fontsize)
        ax_interaction.legend(
            frameon=False, fontsize=legend_fontsize, ncol=len(levels),
            loc="upper left", handlelength=1.3, columnspacing=0.8,
        )

        # Include the interaction curves in the shared effect scale.
        share_effect_limits([*shape_axes, ax_interaction], (ALPHA, BETA, GAMMA))
    else:
        m2_grid = np.arange(int(training_counts[:, 1].max()) + 1)
        if interaction_view == "3d":
            x2_grid = np.linspace(*FEATURE_RANGE, 100, dtype=np.float32)
        else:
            x_edges = np.linspace(*FEATURE_RANGE, 161)
            x2_grid = (x_edges[:-1] + x_edges[1:]) / 2
            m_edges = np.arange(len(m2_grid) + 1) - 0.5
        X2, M2 = np.meshgrid(x2_grid, m2_grid)
        Z_true = interaction_effect(X2, M2)
        Z_learned = component_values(
            x2=X2.ravel(), m2=M2.ravel()
        )["a2*b2"].to_numpy().reshape(X2.shape)
        vmax = max(abs(GAMMA), float(np.abs(Z_true).max()), float(np.abs(Z_learned).max()), 1e-8)
        norm = Normalize(vmin=-vmax, vmax=vmax)
        for ax, values, title in zip(
            interaction_axes, [Z_true, Z_learned],
            [r"(h) True $I_{22}^{(0)}$", r"(i) Learned $I_{22}^{(0)}$"],
        ):
            if interaction_view == "3d":
                ax.plot_surface(
                    X2, M2, values, cmap="RdBu_r", norm=norm,
                    rcount=X2.shape[0], ccount=X2.shape[1], linewidth=0,
                    antialiased=True, shade=False, rasterized=True,
                )
                ax.set(xlim=FEATURE_RANGE, ylim=(m2_grid.min(), m2_grid.max()),
                       zlim=(-1.05 * vmax, 1.05 * vmax))
                ax.set_xlabel(r"$x_2$", fontsize=fontsize, labelpad=-3)
                ax.set_ylabel(r"$m_2$", fontsize=fontsize, labelpad=-3)
                ax.set_title(title, fontsize=fontsize, pad=7)
                ax.view_init(elev=25, azim=-110)
                ax.set_box_aspect((1.3, 1.0, 0.85), zoom=1.15)
                ax.tick_params(labelsize=tick_fontsize, pad=0)
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
                ax.zaxis.set_major_locator(MaxNLocator(3))
            else:
                ax.pcolormesh(x_edges, m_edges, values, cmap="RdBu_r", norm=norm,
                              shading="flat", rasterized=True)
                ax.set(xlabel=r"$x_2$", xlim=FEATURE_RANGE, ylim=(m_edges[0], m_edges[-1]))
                ax.set_title(title, fontsize=fontsize)
                ax.yaxis.set_major_locator(MaxNLocator(3, integer=True))
        if interaction_view == "heatmap":
            interaction_axes[0].set_ylabel(r"Triangle count $m_2$")
            interaction_axes[1].tick_params(labelleft=False)
        cax = fig.add_subplot(bottom[6])
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax)
        colorbar.set_label(r"$I_{22}^{(0)}$", fontsize=fontsize)
        colorbar.ax.tick_params(labelsize=tick_fontsize)
        colorbar.locator = MaxNLocator(3)
        colorbar.update_ticks()

    fig.legend(
        handles=[
            Line2D([], [], color=TRUE_COLOR, lw=shape_linewidth, label="True"),
            Line2D(
                [], [], color=MODEL_COLOR, lw=shape_linewidth,
                ls="--", label="MI-GNAN",
            ),
        ],
        loc="outside lower center", ncol=2, frameon=False, fontsize=legend_fontsize,
        handlelength=2.2, columnspacing=1.5,
    )
    return fig


def plot_diagnostic_overview(data, *, seed=11):
    """Alternative six-panel view of the same test data."""
    effects = data["effects"]
    ALPHA, BETA, GAMMA = effects.alpha, effects.beta, effects.gamma
    FEATURE_RANGE = data["feature_range"]
    feature_effect, path_effect = effects.feature_effect, effects.path_effect
    interaction_effect = effects.interaction_effect
    component_values = data["component_values"]
    training_counts = data["training_counts"]
    true_intercept = 0.0
    gid = data["graph_id"]
    g = data["graph"]
    explanation = data["explanation"]
    true_local, learned_local = data["true_local"], data["pred_local"]
    truth, learned = data["truth"], data["learned"]
    channel_rmse = data["channel_rmse"]

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6.7), constrained_layout=True
    )
    true_color, model_color = TRUE_COLOR, MODEL_COLOR

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    # ---------- Prima riga: recupero dei tre canali sul test ----------
    for k, (ax, name) in enumerate(zip(axes[0], ["F", "S", "I"])):
        t, p = truth[:, k], learned[:, k]
        low, high = min(t.min(), p.min()), max(t.max(), p.max())
        margin = 0.06 * max(high - low, 0.1)
        limits = (low - margin, high + margin)

        ax.scatter(
            t, p, s=8, alpha=0.35, color=model_color,
            edgecolors="none", rasterized=True,
        )
        ax.plot(limits, limits, color=true_color, lw=1.3)
        ax.set(
            xlim=limits, ylim=limits,
            xlabel="True contribution",
            ylabel="Learned contribution",
            title=f"({chr(97 + k)}) Test: {name} · RMSE {channel_rmse[k]:.4f}",
        )
        ax.set_aspect("equal", adjustable="box")

    # ---------- Seconda riga, sinistra: decomposizione del grafo ----------
    ax = axes[1, 0]
    positions = np.arange(3)
    width = 0.36

    ax.bar(
        positions - width / 2, true_local.sum(axis=0),
        width, color=true_color, label="True",
    )
    ax.bar(
        positions + width / 2, learned_local.sum(axis=0),
        width, color=model_color, label="MI-GNAN",
    )
    ax.axhline(0, color="gray", lw=0.7)
    ax.set(
        xticks=positions, xticklabels=["F", "S", "I"],
        ylabel="Graph contribution",
        title=(
            f"(d) Graph {gid}: decomposition\n"
            f"Observed target {g.y:.3f} · prediction {explanation['output']:.3f}"
        ),
    )
    ax.set_xlabel(
        f"Intercept: true {true_intercept:.3f} · "
        f"learned {explanation['intercept']:.3f}",
        fontsize=8,
    )
    ax.legend(frameon=False, fontsize=8)

    # ---------- Seconda riga, centro: contributi dei singoli nodi ----------
    ax = axes[1, 1]
    node_ids = np.arange(g.num_nodes)
    true_node_total = true_local.sum(axis=1)
    learned_node_total = learned_local.sum(axis=1)
    local_rmse = np.sqrt(np.mean(
        (true_node_total - learned_node_total) ** 2
    ))

    ax.bar(
        node_ids - width / 2, true_node_total,
        width, color=true_color,
    )
    ax.bar(
        node_ids + width / 2, learned_node_total,
        width, color=model_color,
    )
    ax.axhline(0, color="gray", lw=0.7)
    ax.set(
        xticks=node_ids,
        xlabel="Node",
        ylabel="F + S + I",
        title=f"(e) Graph {gid}: nodes · RMSE {local_rmse:.4f}",
    )

    # ---------- Seconda riga, destra: shape e input del grafo ----------
    ax = axes[1, 2]
    grid = np.linspace(*FEATURE_RANGE, 250, dtype=np.float32)
    true_curve = feature_effect(grid)
    learned_curve = component_values(x1=grid)["f1"].to_numpy()

    ax.plot(
        grid, true_curve, color=true_color, lw=2,
        label="True",
    )
    ax.plot(
        grid, learned_curve, color=model_color, lw=2,
        ls="--", label="MI-GNAN",
    )

    # I punti evidenziano dove i nodi del grafo interrogano la shape.
    node_x = g.x[:, 0]
    node_f1 = explanation["components"]["f1"].to_numpy()
    ax.scatter(
        node_x, node_f1, s=30, color=model_color,
        edgecolors="white", linewidths=0.6, zorder=4,
        label=f"Nodes of graph {gid}",
    )
    ax.axhline(0, color="gray", lw=0.7)
    ax.set(
        xlabel=r"$x_1$",
        ylabel="Contribution at zero reference",
        title=r"(f) Learned shape $f_1(x_1)$",
    )
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Zero-reference effect recovery: test set and a local explanation · seed {seed}",
        fontsize=13,
    )
    return fig
