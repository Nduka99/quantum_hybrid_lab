"""Matched comparison of classical models and a quantum-kernel SVM on identical inputs.

The protocol was fixed before the reference grid was run (README, "Method"):

- one stratified sample; standard scaler -> PCA (components = qubits) -> min-max [0, 1], all
  fitted on the training rows only;
- logistic regression, linear SVM, RBF SVM and a quantum-kernel SVM see exactly those inputs, are
  tuned on the same folds with the same scorer, and are scored once on the held-out rows;
- the quantum kernel's bandwidth is tuned like RBF gamma, smaller bandwidth winning ties;
- LightGBM on all original features of the same training rows is reported as a full-feature
  classical control. It is not the evidence gate's classical ceiling (full training evidence).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qhcyber.lab.datasets import LabDataset, dataset_summary
from qhcyber.quantum.kernels import (
    build_feature_map,
    circuit_resources,
    fidelity_kernel,
    kernel_diagnostics,
    statevectors,
)

BANDWIDTHS = (0.05, 0.1, 0.25, 0.5, 1.0, float(np.pi))
# Amendment: first-run optima sat on the top grid edge, so both grids were widened (README).
C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
GAMMA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5
SCORING = "balanced_accuracy"

MATCHED_CLASSICAL = "matched classical"
MATCHED_QUANTUM = "matched quantum"
FULL_FEATURE_CONTROL = "full-feature classical control"

SANDBOX_NOTICE = (
    "Sandbox demonstration, not research evidence: one small sample, exact noiseless "
    "simulation, and no claim that the quantum kernel is better or worse in general."
)


@dataclass(frozen=True)
class LabConfig:
    dataset: str
    n_qubits: int
    reps: int = 2
    entanglement: str = "linear"
    n_train: int = 200
    n_test: int = 200
    seed: int = 0
    bandwidths: tuple[float, ...] = BANDWIDTHS


def split_rows(
    labels: np.ndarray, n_train: int, n_test: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint, stratified training and test row indices."""

    train, test = train_test_split(
        np.arange(len(labels)),
        train_size=n_train,
        test_size=n_test,
        stratify=labels,
        random_state=seed,
    )
    return np.sort(train), np.sort(test)


def fit_representation(
    x_train: np.ndarray, x_test: np.ndarray, n_components: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Standardise, project and rescale to [0, 1], fitted on the training rows only."""

    representation = make_pipeline(
        StandardScaler(), PCA(n_components=n_components, random_state=seed), MinMaxScaler()
    )
    return representation.fit_transform(x_train), representation.transform(x_test)


def holdout_metrics(y_true: np.ndarray, predicted: np.ndarray, score: np.ndarray) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
    }


def tune(estimator: BaseEstimator, grid: dict, x: np.ndarray, y: np.ndarray, folds: list):
    return GridSearchCV(estimator, grid, cv=folds, scoring=SCORING).fit(x, y)


def model_record(
    name: str, tier: str, hyperparameters: dict, cv_score: float | None, test: dict, seconds: float
) -> dict[str, Any]:
    return {
        "name": name,
        "tier": tier,
        "hyperparameters": hyperparameters,
        "cv_balanced_accuracy": None if cv_score is None else float(cv_score),
        "test": test,
        "seconds": round(seconds, 3),
    }


def classical_candidates() -> list[tuple[str, BaseEstimator, dict]]:
    return [
        (
            "Logistic regression",
            LogisticRegression(class_weight="balanced", max_iter=5000),
            {"C": C_GRID},
        ),
        ("Linear SVM", SVC(kernel="linear", class_weight="balanced"), {"C": C_GRID}),
        (
            "RBF SVM",
            SVC(kernel="rbf", class_weight="balanced"),
            {"C": C_GRID, "gamma": GAMMA_GRID},
        ),
    ]


def run_classical(z_train, y_train, z_test, y_test, folds) -> list[dict[str, Any]]:
    records = []
    for name, estimator, grid in classical_candidates():
        started = time.perf_counter()
        search = tune(estimator, grid, z_train, y_train, folds)
        test = holdout_metrics(y_test, search.predict(z_test), search.decision_function(z_test))
        records.append(
            model_record(
                name,
                MATCHED_CLASSICAL,
                dict(search.best_params_),
                search.best_score_,
                test,
                time.perf_counter() - started,
            )
        )
    return records


def run_quantum(config: LabConfig, z_train, y_train, z_test, y_test, folds) -> dict[str, Any]:
    """Tune bandwidth and C on training folds, then simulate test states once, at the winner."""

    started = time.perf_counter()
    feature_map = build_feature_map(config.n_qubits, config.reps, config.entanglement)
    curve = []
    best = None
    for bandwidth in sorted(config.bandwidths):
        train_states = statevectors(feature_map, bandwidth * z_train)
        kernel = fidelity_kernel(train_states, train_states)
        search = tune(
            SVC(kernel="precomputed", class_weight="balanced"),
            {"C": C_GRID},
            kernel,
            y_train,
            folds,
        )
        curve.append(
            {
                "bandwidth": bandwidth,
                "cv_balanced_accuracy": float(search.best_score_),
                "C": search.best_params_["C"],
                "offdiagonal_mean": kernel_diagnostics(kernel, y_train)["offdiagonal_mean"],
            }
        )
        # Strictly greater keeps the smaller bandwidth on ties (bandwidths are ascending).
        if best is None or search.best_score_ > best["search"].best_score_:
            best = {"bandwidth": bandwidth, "search": search, "states": train_states, "K": kernel}

    test_kernel = fidelity_kernel(
        statevectors(feature_map, best["bandwidth"] * z_test), best["states"]
    )
    search = best["search"]
    test = holdout_metrics(
        y_test, search.predict(test_kernel), search.decision_function(test_kernel)
    )
    record = model_record(
        "Quantum-kernel SVM",
        MATCHED_QUANTUM,
        {"C": search.best_params_["C"], "bandwidth": best["bandwidth"]},
        search.best_score_,
        test,
        time.perf_counter() - started,
    )
    details = {
        "feature_map": f"ZZ, {config.reps} reps, {config.entanglement} entanglement",
        "bandwidth_curve": curve,
        "selected_bandwidth": best["bandwidth"],
        "kernel_diagnostics": kernel_diagnostics(best["K"], y_train),
        "resources": circuit_resources(
            config.n_qubits,
            config.reps,
            config.entanglement,
            n_train=len(z_train),
            n_test=len(z_test),
            n_bandwidths=len(config.bandwidths),
        ),
    }
    return {"record": record, "details": details}


def run_full_feature_control(
    x_train, y_train, x_test, y_test, feature_names: tuple[str, ...], seed: int
) -> dict[str, Any]:
    started = time.perf_counter()
    hyperparameters = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 10,
    }
    model = LGBMClassifier(
        **hyperparameters, class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1
    )
    # Named frames keep LightGBM and scikit-learn agreeing on the feature contract.
    train = pd.DataFrame(x_train, columns=list(feature_names))
    test_frame = pd.DataFrame(x_test, columns=list(feature_names))
    model.fit(train, y_train)
    test = holdout_metrics(y_test, model.predict(test_frame), model.predict_proba(test_frame)[:, 1])
    return model_record(
        "LightGBM (all features)",
        FULL_FEATURE_CONTROL,
        {**hyperparameters, "tuned": False},
        None,
        test,
        time.perf_counter() - started,
    )


def verdict(models: list[dict[str, Any]], n_test: int) -> str:
    """Plain description of this run. Deliberately has no 'advantage' outcome."""

    quantum = next(model for model in models if model["tier"] == MATCHED_QUANTUM)
    matched = [model for model in models if model["tier"] == MATCHED_CLASSICAL]
    best = max(matched, key=lambda model: model["test"]["balanced_accuracy"])
    difference = quantum["test"]["balanced_accuracy"] - best["test"]["balanced_accuracy"]
    if abs(difference) < 0.01:
        comparison = "within 0.01 of"
    elif difference > 0:
        comparison = f"{difference:.3f} above"
    else:
        comparison = f"{-difference:.3f} below"
    return (
        f"On this sample the quantum-kernel SVM's held-out balanced accuracy was {comparison} "
        f"the best matched classical model ({best['name']}, "
        f"{best['test']['balanced_accuracy']:.3f}). A single run on {n_test} test rows cannot "
        "separate that from sampling noise; the reference runs repeat it over five seeds."
    )


def run_matched_benchmark(dataset: LabDataset, config: LabConfig) -> dict[str, Any]:
    """Run every model on one shared sample and return a JSON-compatible report."""

    started = time.perf_counter()
    train_rows, test_rows = split_rows(dataset.labels, config.n_train, config.n_test, config.seed)
    x_train, x_test = dataset.features[train_rows], dataset.features[test_rows]
    y_train, y_test = dataset.labels[train_rows], dataset.labels[test_rows]
    z_train, z_test = fit_representation(x_train, x_test, config.n_qubits, config.seed)
    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=config.seed)
    folds = list(splitter.split(z_train, y_train))

    models = run_classical(z_train, y_train, z_test, y_test, folds)
    quantum = run_quantum(config, z_train, y_train, z_test, y_test, folds)
    models.append(quantum["record"])
    models.append(
        run_full_feature_control(
            x_train, y_train, x_test, y_test, dataset.feature_names, config.seed
        )
    )

    rbf = next(model for model in models if model["name"] == "RBF SVM")
    rbf_matrix = rbf_kernel(z_train, gamma=rbf["hyperparameters"]["gamma"])
    return {
        "config": {**asdict(config), "bandwidths": list(config.bandwidths)},
        "dataset": dataset_summary(dataset),
        "sample": {
            "train_rows": int(len(train_rows)),
            "test_rows": int(len(test_rows)),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "representation": (
                f"standard scaler -> PCA({config.n_qubits}) -> min-max [0, 1], fitted on the "
                "training rows only"
            ),
            "tuning": f"{N_FOLDS}-fold stratified CV shared by every matched model; {SCORING}",
        },
        "models": models,
        "quantum": {
            **quantum["details"],
            "rbf_kernel_diagnostics": kernel_diagnostics(rbf_matrix, y_train),
        },
        "verdict": verdict(models, len(test_rows)),
        "notice": SANDBOX_NOTICE,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
