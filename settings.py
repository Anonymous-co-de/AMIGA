"""Dataset schemas and editable defaults. Columns refer to the raw node matrix."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parents[1] / "data" / "dataset"
CACHE_ROOT = ROOT / "motif_counts"


@dataclass(frozen=True)
class Feature:
    """One logical variable: numerical column, category ID, or entire one-hot group.

    Category IDs must be integers in [0, len(categories)). For one-hot input,
    columns and categories have the same order. Numerical columns are scaled
    on training nodes; categorical values are never standardized.
    """

    name: str
    kind: str
    columns: tuple[int, ...]
    categories: tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind not in {"numerical", "categorical", "one_hot"}:
            raise ValueError(f"Unknown feature kind: {self.kind}")
        if not self.columns or len(set(self.columns)) != len(self.columns) or min(self.columns) < 0:
            raise ValueError("Feature columns must be distinct nonnegative indices")
        if self.kind != "one_hot" and len(self.columns) != 1:
            raise ValueError("Numerical/category-ID features require exactly one column")
        if self.kind == "numerical" and self.categories:
            raise ValueError("Numerical features cannot have categories")
        if self.kind != "numerical" and not self.categories:
            raise ValueError("Declare the allowed categories explicitly")
        if self.kind == "one_hot" and len(self.columns) != len(self.categories):
            raise ValueError("One-hot columns must match the number of categories")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("Category names must be unique")


ATOM_TYPES = ("C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca")
QM9_FEATURES = (
    Feature("atom", "one_hot", tuple(range(5)), ("H", "C", "N", "O", "F")),
    Feature("atomic_number", "numerical", (5,)),
    *(Feature(name, "categorical", (column,), ("no", "yes"))
      for column, name in enumerate(("aromatic", "sp", "sp2", "sp3"), start=6)),
    Feature("num_hydrogens", "numerical", (10,)),
)

# Category names below are column IDs when the source does not provide a
# verified mapping to atom names. Feature kinds are explicit, never inferred.
DATASETS = {
    "mutagenicity": {
        "name": "Mutagenicity",
        "loader": "tu",
        "path": DATA_ROOT / "Mutagenicity",
        "task": "classification",
        "target": "nonmutagen",
        "features": (Feature("atom", "one_hot", tuple(range(14)), ATOM_TYPES),),
        "training": {"batch_size": 256},
    },
    "nci1": {
        "name": "NCI1", "loader": "tu", "path": DATA_ROOT / "NCI1",
        "task": "classification", "target": "activity",
        "features": (Feature("atom", "one_hot", tuple(range(37)),
                             tuple(str(i) for i in range(37))),),
        "training": {"batch_size": 256},
    },
    "proteins": {
        "name": "PROTEINS", "loader": "tu", "path": DATA_ROOT / "PROTEINS",
        "task": "classification", "target": "protein_class",
        "features": (
            Feature("node_attribute", "numerical", (0,)),
            Feature("node_label", "one_hot", (1, 2, 3), ("0", "1", "2")),
        ),
        "training": {"batch_size": 256},
    },
    "ptc_mr": {
        "name": "PTC_MR", "loader": "tu", "path": DATA_ROOT / "PTC_MR",
        "task": "classification", "target": "carcinogenicity",
        "features": (Feature("atom", "one_hot", tuple(range(18)),
                             tuple(str(i) for i in range(18))),),
        "training": {"batch_size": 256},
    },
    **{
        key: {
            "name": name, "loader": "pickle", "path": DATA_ROOT / f"{key}.pkl",
            "task": "regression", "target": target,
            "features": tuple(Feature(f"x{i + 1}", "numerical", (i,)) for i in range(10)),
            "training": {"epochs": 500, "patience": 100, "batch_size": batch_size},
        }
        for key, name, target, batch_size in (
            ("bareg1", "BAReg1", "motif_volume", 256),
            ("bareg2", "BAReg2", "motif_count", 32),
        )
    },
    "crippen": {
        "name": "Crippen", "loader": "pickle", "path": DATA_ROOT / "crippen.pkl",
        "task": "regression", "target": "crippen_logp",
        "features": (Feature("atom", "one_hot", tuple(range(14)),
                             tuple(str(i) for i in range(14))),),
        "training": {"epochs": 500, "patience": 100, "batch_size": 256},
    },
    **{
        f"qm9_{target}": {
            "name": f"QM9-{target}", "loader": "qm9", "path": DATA_ROOT / "qm9",
            "task": "regression", "target": target, "target_index": index,
            "features": QM9_FEATURES,
            "motifs": {"sizes": (3, 4, 5), "null_replicates": 500, "mining_graphs": None},
            "training": {"epochs": 250, "patience": 50, "batch_size": 256},
        }
        for index, target in enumerate(("mu", "alpha", "homo"))
    },
}

# Example for custom mixed data:
# features = (Feature("charge", "numerical", (0,)),
#             Feature("atom", "categorical", (1,), ("C", "N", "O")))

MOTIF_SETTINGS = dict(sizes=(3, 4, 5), vocabulary_mode="enriched",
                      null_replicates=500, rewires_per_edge=60, fdr=0.05,
                      max_motifs=None, seed=31415, jobs=8)
MODEL_SETTINGS = dict(hidden=16, depth=2, dropout=0.0)
TRAIN_SETTINGS = dict(epochs=300, patience=45, batch_size=128, lr=1e-3,
                      weight_decay=1e-5, lambda_sparse=1e-3, seed=11,
                      device="auto")
