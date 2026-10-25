"""The lab benchmark is leakage-safe, matched across models and honest in its wording."""

import json
from pathlib import Path

import numpy as np
import pytest

from qhcyber.lab.benchmark import (
    FULL_FEATURE_CONTROL,
    MATCHED_CLASSICAL,
    MATCHED_QUANTUM,
    LabConfig,
    fit_representation,
    run_matched_benchmark,
    split_rows,
)
from qhcyber.lab.datasets import DATASET_KEYS, load_dataset

ASSET_DIR = Path(__file__).resolve().parents[2] / "deployment" / "lab" / "assets"


@pytest.fixture(scope="module")
def report() -> dict:
    dataset = load_dataset("breast_cancer", ASSET_DIR)
    config = LabConfig(dataset="breast_cancer", n_qubits=2, reps=1, n_train=60, n_test=40)
    return run_matched_benchmark(dataset, config)


@pytest.mark.parametrize("key", DATASET_KEYS)
def test_datasets_load_with_binary_labels(key) -> None:
    dataset = load_dataset(key, ASSET_DIR)
    assert dataset.features.shape == (len(dataset.labels), len(dataset.feature_names))
    assert set(np.unique(dataset.labels)) == {0, 1}
    assert np.isfinite(dataset.features).all()


def test_rows_are_disjoint_stratified_and_reproducible() -> None:
    labels = np.array([0] * 70 + [1] * 30)
    train, test = split_rows(labels, 40, 20, seed=3)
    assert not set(train) & set(test)
    assert labels[train].mean() == pytest.approx(0.3)
    again_train, again_test = split_rows(labels, 40, 20, seed=3)
    assert np.array_equal(train, again_train) and np.array_equal(test, again_test)


def test_representation_is_fitted_on_training_rows_only() -> None:
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(50, 6))
    z_train, z_test = fit_representation(x_train, rng.normal(size=(20, 6)), 3, seed=0)
    z_train_again, _ = fit_representation(x_train, 100 + rng.normal(size=(20, 6)), 3, seed=0)
    np.testing.assert_array_equal(z_train, z_train_again)
    np.testing.assert_allclose(z_train.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z_train.max(axis=0), 1.0, atol=1e-12)
    assert z_test.shape == (20, 3)


def test_report_is_json_compatible_and_complete(report) -> None:
    assert json.loads(json.dumps(report)) == report
    tiers = [model["tier"] for model in report["models"]]
    assert tiers.count(MATCHED_CLASSICAL) == 3
    assert tiers.count(MATCHED_QUANTUM) == 1
    assert tiers.count(FULL_FEATURE_CONTROL) == 1
    for model in report["models"]:
        assert all(0.0 <= value <= 1.0 for value in model["test"].values())
    assert report["sample"]["train_rows"] == 60 and report["sample"]["test_rows"] == 40


def test_bandwidth_is_tuned_over_the_preregistered_grid(report) -> None:
    curve = report["quantum"]["bandwidth_curve"]
    assert [point["bandwidth"] for point in curve] == sorted(LabConfig("x", 2).bandwidths)
    best = max(point["cv_balanced_accuracy"] for point in curve)
    first_best = next(p["bandwidth"] for p in curve if p["cv_balanced_accuracy"] == best)
    assert report["quantum"]["selected_bandwidth"] == first_best


def test_fixed_bandwidth_skips_tuning() -> None:
    dataset = load_dataset("breast_cancer", ASSET_DIR)
    config = LabConfig("breast_cancer", 2, reps=1, n_train=40, n_test=40, bandwidths=(np.pi,))
    report = run_matched_benchmark(dataset, config)
    assert report["quantum"]["selected_bandwidth"] == pytest.approx(np.pi)
    assert len(report["quantum"]["bandwidth_curve"]) == 1


def test_wording_never_claims_advantage(report) -> None:
    text = (report["verdict"] + report["notice"]).lower()
    assert "advantage" not in text
    assert "not research evidence" in text
