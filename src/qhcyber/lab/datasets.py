"""Datasets available to the hybrid lab, each with its field, licence and citation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

NETFLOW_SAMPLE_FILE = "network_flows_sample.csv.gz"


@dataclass(frozen=True)
class LabDataset:
    key: str
    title: str
    field: str
    positive_class: str
    features: np.ndarray  # (rows, columns), float64
    labels: np.ndarray  # (rows,), 1 = positive class
    feature_names: tuple[str, ...]
    preparation: str
    citation: str
    licence: str


def load_network_flows(asset_dir: Path) -> LabDataset:
    frame = pd.read_csv(asset_dir / NETFLOW_SAMPLE_FILE)
    names = tuple(column for column in frame.columns if column not in ("label", "attack_family"))
    # Raw NetFlow counts span many orders of magnitude; log1p is fixed and label-free.
    features = np.log1p(frame.loc[:, list(names)].to_numpy(dtype=np.float64))
    return LabDataset(
        key="network_flows",
        title="Network flows (NF-UNSW-NB15-v2)",
        field="Cybersecurity",
        positive_class="attack flow",
        features=features,
        labels=frame["label"].to_numpy(dtype=np.int64),
        feature_names=names,
        preparation=(
            "25 shortcut-resistant behaviour features from the model_fit partition; duplicate and "
            "label-conflicting vectors removed; 1,000 benign and 1,000 attack flows; log1p."
        ),
        citation=(
            "Sarhan, Layeghy & Portmann (2022). Towards a Standard Feature Set for Network "
            "Intrusion Detection System Datasets. https://doi.org/10.1007/s11036-021-01843-0"
        ),
        licence=(
            "Academic re-use with citation; commercial use restricted (see the README data notice)."
        ),
    )


def load_breast_cancer_dataset() -> LabDataset:
    bundle = load_breast_cancer()
    return LabDataset(
        key="breast_cancer",
        title="Breast Cancer Wisconsin (Diagnostic)",
        field="Medicine",
        positive_class="malignant tumour",
        features=np.asarray(bundle.data, dtype=np.float64),
        # scikit-learn codes malignant as 0; the lab treats the class of interest as 1.
        labels=(bundle.target == 0).astype(np.int64),
        feature_names=tuple(bundle.feature_names),
        preparation="All 569 samples and 30 cell-nucleus features, as bundled with scikit-learn.",
        citation=(
            "Wolberg, Mangasarian, Street & Street (1993). Breast Cancer Wisconsin (Diagnostic). "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B"
        ),
        licence="CC BY 4.0",
    )


def load_dataset(key: str, asset_dir: Path) -> LabDataset:
    if key == "network_flows":
        return load_network_flows(asset_dir)
    if key == "breast_cancer":
        return load_breast_cancer_dataset()
    raise ValueError(f"Unknown dataset {key!r}.")


DATASET_KEYS = ("network_flows", "breast_cancer")


def dataset_summary(dataset: LabDataset) -> dict[str, Any]:
    """JSON-compatible description, without the data itself."""

    return {
        "key": dataset.key,
        "title": dataset.title,
        "field": dataset.field,
        "positive_class": dataset.positive_class,
        "rows": int(dataset.features.shape[0]),
        "features": int(dataset.features.shape[1]),
        "positive_rate": float(dataset.labels.mean()),
        "preparation": dataset.preparation,
        "citation": dataset.citation,
        "licence": dataset.licence,
    }
