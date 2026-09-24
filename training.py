"""Sparse optimization, validation checkpoint selection, evaluation, and persistence."""

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, mean_absolute_error, r2_score

from .data import Preprocessor, loader
from .model import MIGNAN
from .settings import Feature


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def task_loss(output, targets, task):
    if task == "regression":
        return F.mse_loss(output[:, 0], targets)
    if output.shape[1] == 1:
        return F.binary_cross_entropy_with_logits(output[:, 0], targets)
    return F.cross_entropy(output, targets.long())


@torch.no_grad()
def evaluate(model, batches, preprocessing):
    """Return metrics and predictions in original units/label encoding."""
    model.eval()
    outputs, targets, ids = [], [], []
    for batch in batches:
        batch = batch.to(model.intercept.device)
        outputs.append(model(batch).cpu())
        targets.append(batch.y.cpu())
        ids.append(batch.graph_ids.cpu())
    if not outputs:
        raise ValueError("Evaluation requires at least one graph")
    output, y = torch.cat(outputs), torch.cat(targets)
    metrics = {"loss": float(task_loss(output, y, preprocessing.task))}
    if preprocessing.task == "regression":
        prediction = output[:, 0].numpy() * preprocessing.target_scale + preprocessing.target_mean
        target = y.numpy() * preprocessing.target_scale + preprocessing.target_mean
        metrics.update(rmse=float(np.sqrt(np.mean((prediction - target) ** 2))),
                       mae=float(mean_absolute_error(target, prediction)),
                       r2=float(r2_score(target, prediction)) if len(target) > 1 else float("nan"))
        result = dict(predictions=prediction, targets=target)
    else:
        if output.shape[1] == 1:
            p = output[:, 0].sigmoid().numpy()
            probabilities = np.column_stack([1 - p, p])
        else:
            probabilities = output.softmax(1).numpy()
        encoded = probabilities.argmax(1)
        y = y.numpy().astype(int)
        metrics.update(accuracy=float(accuracy_score(y, encoded)),
                       balanced_accuracy=float(balanced_accuracy_score(y, encoded)))
        if len(preprocessing.classes) == 2:
            metrics["auroc"] = float(roc_auc_score(y, probabilities[:, 1])) if len(np.unique(y)) == 2 else float("nan")
        result = dict(predictions=preprocessing.classes[encoded], targets=preprocessing.classes[y],
                      probabilities=probabilities)
    return metrics, dict(graph_ids=torch.cat(ids).numpy(), outputs=output.numpy(), **result)


@dataclass
class FitResult:
    model: MIGNAN
    history: pd.DataFrame
    best_epoch: int
    best_validation: float


def fit(model, records, train_indices, val_indices, preprocessing, *, epochs=300, patience=45,
        batch_size=128, lr=1e-3, weight_decay=1e-5, lambda_sparse=1e-3,
        seed=11, device="auto", verbose=True):
    """Train the supplied model in place; restore its best validation checkpoint.

    Centering is refreshed before training and after each epoch, without gradients.
    Within an epoch its buffers are fixed. The sparsity estimate uses each shuffled
    minibatch's graph-balanced means and the same forward/dropout draw as the loss.
    Binary selection uses AUROC, multiclass cross-entropy, regression RMSE.
    Seed the model constructor separately to reproduce parameter initialization.
    """
    if epochs < 1 or patience < 1 or batch_size < 1 or lambda_sparse < 0:
        raise ValueError("Invalid training settings")
    train_indices, val_indices = np.asarray(train_indices), np.asarray(val_indices)
    for indices in (train_indices, val_indices):
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(indices) == 0 or
                len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(records)):
            raise ValueError("Invalid training/validation indices")
    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("Training and validation sets must be disjoint")
    if model.config["outputs"] != preprocessing.outputs or model.schema != preprocessing.schema:
        raise ValueError("Model must match the preprocessing schema and output count")
    seed_everything(seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model.to(device)
    train_batches = loader(records, train_indices, batch_size, True)
    center_batches = loader(records, train_indices, batch_size)
    val_batches = loader(records, val_indices, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    binary = preprocessing.task == "classification" and preprocessing.outputs == 1
    metric = "auroc" if binary else "rmse" if preprocessing.task == "regression" else "loss"
    best = -np.inf if binary else np.inf
    state, best_epoch, stale, history = None, 0, 0, []
    model.refresh_centering(center_batches)
    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(3)
        for batch in train_batches:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output, parts = model(batch, return_components=True)
            task = task_loss(output, batch.y, preprocessing.task)
            penalty = model.sparsity(parts, batch)
            loss = task + lambda_sparse * penalty
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            totals += np.array([task.item(), penalty.item(), loss.item()]) * batch.num_graphs
        model.refresh_centering(center_batches)
        validation, _ = evaluate(model, val_batches, preprocessing)
        score = validation[metric]
        if not np.isfinite(score):
            raise ValueError(f"Validation {metric} is undefined; check labels/split")
        history.append(dict(epoch=epoch, **dict(zip(("train_task", "train_sparsity", "train_loss"),
                                                    totals / len(train_indices))),
                            **{f"val_{k}": v for k, v in validation.items()}))
        improved = score > best + 1e-6 if binary else score < best - 1e-6
        if improved:
            best, best_epoch, stale = score, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"Epoch {epoch:3d} | task={history[-1]['train_task']:.4f} | "
                  f"sparsity={history[-1]['train_sparsity']:.4f} | val_{metric}={score:.4f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(state)
    model.eval()
    return FitResult(model, pd.DataFrame(history), best_epoch, float(best))


def save_model(path, model, preprocessing, vocabulary):
    """Save everything needed for inference: shapes, reference, schema, and motifs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prep = asdict(preprocessing)
    for key, value in prep.items():
        if isinstance(value, np.ndarray):
            prep[key] = value.tolist()
    torch.save(dict(config=model.config, preprocessing=prep, vocabulary=vocabulary.to_json(orient="records"),
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), path)


def load_model(path, device="cpu"):
    """Return model, preprocessing, and ordered vocabulary for new-graph inference."""
    import json

    stored = torch.load(path, map_location="cpu", weights_only=True)
    prep = stored["preprocessing"]
    prep["schema"] = tuple(Feature(**f) for f in prep["schema"])
    for key in ("feature_mean", "feature_scale", "topology_mean", "topology_scale", "classes"):
        prep[key] = np.asarray(prep[key])
    preprocessing = Preprocessor(**prep)
    model = MIGNAN(preprocessing.schema, **stored["config"]).to(device)
    model.load_state_dict(stored["state_dict"])
    return model.eval(), preprocessing, pd.DataFrame(json.loads(stored["vocabulary"]))
