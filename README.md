# AMIGA

Standalone AMIGA implementation:

```text
score(G) = intercept + sum_over_nodes(F + S + A * B)
```

Rank 1 per logit, sum pooling, MLPs for numerical variables, and lookup tables
for categorical variables. Each one-hot group is a single variable. The four
function banks have separate parameters. With multiple logits, each shape's
MLP shares its hidden layers across outputs, while outputs and products remain
separate.

## Getting started

```bash
conda activate gnan-clean
```

Open **`usage.ipynb`** and select the interpreter from the `gnan-clean` environment.
Set `DATASET` in the first cell and run sections 1–7: loading, motifs,
preprocessing, training, evaluation, and saving. `MAX_GRAPHS=None` uses all
graphs; an integer selects a reproducible subset. The dataset's `training`
and `motifs` options override the shared defaults in `settings.py`.
Results are saved to `results/<dataset>/`. Use `explore.ipynb` to load a saved
model and inspect its learned functions and explanations.

For a quick check, limit the number of graphs, set
`motif_options.update(sizes=(3,), vocabulary_mode="all", null_replicates=0)`
before `build_topology`, and reduce `train_options["epochs"]` before `fit`.
Full mining can take time; subsequent runs reuse the cache. These checks
and default settings are not benchmark results.

If needed, install dependencies with `python -m pip install -r requirements.txt`.
The code requires Python 3.10+ and the igraph API `rewire(..., allowed_edge_types="simple")`.

## Files

| File | Contents |
| --- | --- |
| `settings.py` | Paths, feature schemas, model settings, and training settings |
| `data.py` | Simple undirected graphs, TU/pickle/QM9 loaders, splits, preprocessing, and batches |
| `motifs.py` | Mining, cached candidate/null counts, selection, and lazy occurrences |
| `prepare_counts.py` | Generate and verify reusable archives for all configured datasets |
| `motif_counts/` | Compressed candidate archives and checksum manifest |
| `model.py` | Shapes, lookups, centering, forward pass, and sparsity penalty |
| `training.py` | Fitting, evaluation, saving, and loading |
| `explain.py` | Exact decompositions, occurrence allocations, and figures |
| `usage.ipynb` | Step-by-step training for all datasets |
| `tune.py` | Grid search across configurable seeds, validation CSV, and experiment manifests |
| `explore.ipynb` | Load checkpoints and explore numerical/categorical features and motifs |

Imports use `from mignan...`: add `github/` to the Python path, as in the
notebook's first cell, or run Python from `github/`.
After renaming the package, restart the kernel before rerunning the notebook.
The package does not import modules from the previous implementation.

## Supported datasets

The registry in `settings.py` includes the following ten datasets.

| Key | Task | Logical features |
| --- | --- | --- |
| `mutagenicity` | Classification | Atom type, lookup with 14 categories |
| `nci1` | Classification | Atom type, lookup with 37 categories |
| `proteins` | Classification | One numerical attribute + a label with 3 categories |
| `ptc_mr` | Classification | Atom type, lookup with 18 categories |
| `bareg1`, `bareg2` | Regression | 10 numerical variables |
| `crippen` | Regression | Atom type, lookup with 14 categories |
| `qm9_mu`, `qm9_alpha`, `qm9_homo` | Regression | Atom type, 4 categorical binary indicators, numerical atomic number and hydrogen count |

Schemas match the columns in the local data sources: TU with
`use_node_attr=True`, pickles containing `(adjacency, features, targets)`,
and PyG's QM9. Numerical attributes and redundant atom-type information in
QM9 are retained. Categories without a verified chemical mapping are named
using column IDs rather than atom symbols. Edge attributes are not model
inputs. QM9 targets use the units provided by the PyG loader.

## Custom data and schemas

```python
from mignan import Feature
from mignan.data import Graph

schema = (
    Feature("charge", "numerical", (0,)),
    Feature("atom", "one_hot", (1, 2, 3), ("C", "N", "O")),
    Feature("state", "categorical", (4,), ("off", "on")),
)
# Each x matrix has five columns; the model receives three variables.
# The state column contains IDs 0/1, without one-hot encoding.
graphs = [Graph(x=x, edge_index=edge_index, y=target)]
```

The schema must cover every column exactly once. Allowed categories are
explicitly declared; their vocabulary is not learned from the test set.
Unknown values or invalid one-hot groups raise an error. Custom data can
also be provided directly as a list of `Graph` objects.

`Preprocessor.fit` receives only training indices. Numerical features and
`log1p(counts)` are standardized over training nodes. For regression, the
target is standardized over training graphs. Classification supports scalar
numerical labels, with a mapping saved in `preprocessing.classes`: one logit
for binary classification and one logit per class for multiclass classification.

## Cache

`build_topology` saves lossless, LZMA-compressed NPZ archives in `motif_counts/`.
Each archive contains **all connected candidate motifs**, including motifs not
selected at the current FDR threshold:

- Exact per-node counts for every graph, stored in the smallest fitting unsigned
  integer type (8, 16, 32, or 64 bits).
- Exact total counts for each null-model replicate and candidate.
- Observed mining totals, null mean/standard deviation, z-scores, p-values,
  Benjamini–Hochberg q-values, and motif templates.
- Graph offsets, exact training indices, structural fingerprint, mining settings,
  algorithm version, and igraph version. No features, targets, or pickled objects.

The archive key includes graph structure/order, training membership, motif sizes,
null-model configuration, mining seed/limit, and algorithm/library versions.
It excludes `fdr` (α), `max_motifs`, `vocabulary_mode`, and `jobs`. Changing α
reselects the vocabulary **without repeating counts or null-model generation**:

```python
options = {**MOTIF_SETTINGS, **DATASETS[DATASET].get("motifs", {})}
motifs = build_topology(graphs, train_idx, **{**options, "fdr": 0.10})
```

Refit preprocessing and the model after changing the selected vocabulary.
The null replicate count still determines empirical p-value resolution; raising
that count requires a new archive. Occurrences are enumerated lazily when you
access `motifs.occurrences[graph_id]`, and are not saved to disk.

- `force_recompute=True`: recompute and atomically replace the matching archive.
- `cache_dir=...`: choose the directory; `cache_dir=None` disables caching.
- `mining_graphs=5000`: limit discovery/null statistics to a deterministic subset
  of training graphs (the QM9 default). Candidate counts cover **every graph**;
  scaling, centering, and training use the entire training set.

Committed archives and their checksums are listed in `motif_counts/index.json`.
They use complete datasets, the default split seed (2027), and mining settings
from `settings.py`. The three QM9 targets share one archive. Matching runs reuse
these files automatically; different graph order, splits, settings, or igraph
versions generate separate archives. Old selected-vocabulary caches in `cache/`
are superseded; existing model checkpoints remain usable.

To generate or verify the archives, without training:

```bash
python prepare_counts.py                 # All configured datasets; resumes matching files
python prepare_counts.py ptc_mr proteins # Selected datasets
```

Saving uses a temporary file followed by atomic replacement. Archives are written
before vocabulary selection, so an empty selection still leaves reusable counts.
An unreadable archive raises an error; explicitly regenerate it if needed.

Mining enumerates **induced, connected, unlabeled motifs** of orders 3–6.
Enrichment uses degree-preserving rewiring, one-sided empirical p-values,
and Benjamini–Hochberg correction across all candidates. Selection also
requires the observed count to exceed the null mean. An empty vocabulary
raises an error. The model does not use edge attributes or distinguish
node roles within a motif.

## Training and centering

`fit` optimizes the task loss plus `lambda_sparse * Omega`, using AdamW and
gradient clipping at 5. `Omega` sums the absolute values of all centered
shapes, averaging over nodes within each graph and then over graphs. With
multiple logits, it also averages over outputs. `lambda_sparse=0` disables
this penalty. The penalty encourages small responses without pruning or
hard thresholding to zero.

Centering means use the same equal weighting of graphs. They are recomputed
over the entire training set before training and after each epoch, with
dropout and gradients disabled. They remain fixed during the epoch's
updates: gradients do not propagate through the centering computation.
Refreshing these means can change the model output and is part of the protocol.

Checkpoint selection uses AUROC for binary classification, cross-entropy for
multiclass classification, and RMSE in original units for regression. The
best checkpoint includes the centering means. `fit` returns the model and
training history; the test set is evaluated separately with `evaluate`.
Call `seed_everything` **before constructing the model** to make initialization
reproducible as well.

## Explanations and inference

`explain_graph` returns contributions by node, variable, motif, pair, and
occurrence. For each motif, it subtracts the response at zero participation
and distributes the difference among occurrences incident to the node.
Zero-participation terms are returned separately in `zero_motifs` and
`zero_pairs`. The `pairs` entries in the occurrence table follow the variable
order in the schema. The function verifies count and contribution reconstruction.

Explanations use logit units for classification and original target units
for regression. An absent motif can contribute relative to the reference.
Allocations are not causal deletion effects. Centering defines the channel
separation relative to the product of the marginals; it neither removes
observed dependencies between attributes and topology nor uniquely identifies
the individual factors.

`save_model` preserves weights, schema, scalers, centering, and vocabulary.
After `load_model`, use `count_participation(new_graphs, vocabulary)` and
`preprocessing.inputs(graph, counts)` for new graphs: a target is not required
to transform inputs or call `explain_graph`. The vocabulary remains fixed.

## Hyperparameter tuning

The search grid and training seeds are at the top of `tune.py`.
Every configuration runs on all configured seeds. Dataset-specific epoch,
patience, and batch-size defaults still apply. Run from the package directory:

```bash
conda activate gnan-clean
python tune.py ptc_mr
# Use any nonempty list of distinct seeds:
python tune.py ptc_mr --seeds 11 22 33
# Or choose a new output directory and limit training time:
python tune.py proteins --epochs 100 --output results/tuning/proteins/run_01
```

All configurations use the same train/validation/test split, preprocessing, and
cached motif vocabulary. Seeds vary initialization, minibatch order,
and dropout; they do not define separate folds or independently sampled splits.
Each fit restores its best validation epoch. The winning configuration maximizes
mean validation AUROC for binary classification, or minimizes mean validation
RMSE/cross-entropy for regression/multiclass classification. Ties retain the
first configuration in grid order. The test set is evaluated only for the
winning models, after hyperparameter selection.

Only these files are saved:

```text
results/tuning/<dataset>/<run>/
    run.json       # Search grid, winning parameters/scores, seeds, data/code
                   # fingerprints, resolved settings, environment versions.
    split.npz      # Exact train/validation/test indices in the loaded graph order.
    search_results.csv # Validation results for every configuration and seed.
    seed_<seed>.pt # One winning checkpoint per seed: weights, centering,
                   # schema, scalers, and motif vocabulary.
```

The CSV has one row per configuration/seed, including resolved model/training
parameters, best epoch, all validation metrics at that epoch, and a `selected`
flag identifying the winning configuration. `selection_metric` identifies the
score used for ranking; `validation_mean` and `validation_std` summarize that
score across seeds for each configuration. Standard deviation uses `ddof=0`
and is zero for a single seed. Undefined metrics are empty CSV cells. Test
metrics are recorded only for winners in `run.json`.

The manifest includes the selected epoch, validation score, and final test
metrics for each saved seed. Losing models, epoch histories, optimizer states,
and per-graph predictions are not saved. The existing motif cache is shared
across trials and runs rather than copied into each result directory. Existing
run directories are never overwritten. Checkpoints support inference; they do
not resume an interrupted optimizer session.

`usage.ipynb` follows the same minimal saving policy for a single model:
`model.pt`, `split.npz`, and `run.json`. Environment versions, data fingerprints,
seeds, and resolved parameters identify the experiment; replay requires the
matching data and source code. Exact numerical equality across different
hardware or CUDA/library versions is not guaranteed.

`--seeds` accepts one or more distinct seeds. Other options are listed by
`python tune.py --help`.

## Exploring a saved model

Open `explore.ipynb` with the `gnan-clean` kernel. Set `RUN_DIR` to a saved run,
or leave it as `None` to select the most recent tuning run under
`results/tuning/<dataset>/<run>/`. Choose `SEED` and, for multiclass models,
`OUTPUT_INDEX`. No training is performed.

The notebook displays categorical lookup tables or numerical feature curves,
motif structures and functions, and an exact local decomposition for a graph
from the saved split. It verifies the dataset fingerprint, reuses a matching
motif cache when available, and otherwise counts only the saved vocabulary on
the selected graph. It never reruns motif discovery and does not write new
results, histories, or figures to disk.
