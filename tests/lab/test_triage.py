"""Triage serving reproduces the frozen research model and enforces its input contract."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from qhcyber.serving.assets import verify_manifest
from qhcyber.serving.triage import (
    MAX_FLOWS_PER_REQUEST,
    calibrated_risk,
    load_triage_model,
    score_flows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = PROJECT_ROOT / "deployment" / "lab" / "assets"
RAW_PATH = PROJECT_ROOT / "data/raw/uq_netflow_v2_pilot/NF-UNSW-NB15-v2.csv"
PARTITION_PATH = (
    PROJECT_ROOT / "data/processed/uq_netflow_v2_pilot/benchmark_partition_manifest.csv.gz"
)


@pytest.fixture(scope="module")
def model():
    return load_triage_model(ASSET_DIR)


@pytest.fixture(scope="module")
def examples() -> list[dict]:
    return json.loads((ASSET_DIR / "example_flows.json").read_text(encoding="utf-8"))


def test_assets_match_manifest_hashes() -> None:
    manifest = verify_manifest(ASSET_DIR)
    assert {"lightgbm_behaviour_only.txt", "example_flows.json"} <= set(manifest["files"])


def test_tampered_asset_is_rejected(tmp_path) -> None:
    shutil.copytree(ASSET_DIR, tmp_path / "assets")
    (tmp_path / "assets" / "example_request.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="example_request.json"):
        verify_manifest(tmp_path / "assets")


def test_examples_reproduce_reference_risk(model, examples) -> None:
    scored = score_flows(model, [example["flow"] for example in examples], "strict_fpr_0.001")
    risk = np.array([result["risk"] for result in scored["results"]])
    expected = np.array([example["expected_risk"] for example in examples])
    np.testing.assert_allclose(risk, expected, rtol=0, atol=1e-12)
    alerts = [result["alert"] for result in scored["results"]]
    assert alerts == [value >= scored["threshold"] for value in expected]


def test_policy_changes_only_the_threshold(model, examples) -> None:
    flows = [example["flow"] for example in examples[:3]]
    strict = score_flows(model, flows, "strict_fpr_0.001")
    lenient = score_flows(model, flows, "default_0.5")
    assert strict["threshold"] > lenient["threshold"]
    assert [r["risk"] for r in strict["results"]] == [r["risk"] for r in lenient["results"]]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda flow: flow.pop("IN_BYTES"), "missing"),
        (lambda flow: flow.update({"IPV4_SRC_ADDR": 1.0}), "unexpected"),
        (lambda flow: flow.update({"IN_BYTES": -1.0}), "non-negative"),
        (lambda flow: flow.update({"IN_BYTES": float("nan")}), "finite"),
        (lambda flow: flow.update({"IN_BYTES": 1e39}), "finite"),
    ],
)
def test_rejects_flows_outside_the_contract(model, examples, change, message) -> None:
    flow = dict(examples[0]["flow"])
    change(flow)
    with pytest.raises(ValueError, match=message):
        score_flows(model, [flow], "strict_fpr_0.001")


def test_rejects_bad_batch_size_and_policy(model, examples) -> None:
    flow = examples[0]["flow"]
    with pytest.raises(ValueError, match="between"):
        score_flows(model, [], "strict_fpr_0.001")
    with pytest.raises(ValueError, match="between"):
        score_flows(model, [flow] * (MAX_FLOWS_PER_REQUEST + 1), "strict_fpr_0.001")
    with pytest.raises(ValueError, match="Unknown policy"):
        score_flows(model, [flow], "most_alerts")


@pytest.mark.skipif(
    not (RAW_PATH.exists() and PARTITION_PATH.exists()),
    reason="local research data is not present",
)
def test_full_validation_parity_with_notebook_15(model) -> None:
    """Strict-policy confusion matrix asserted by scripts/build_15 on 478,055 validation flows."""

    pd = pytest.importorskip("pandas")
    dtype = {name: "float32" for name in model.features} | {"Label": "int8"}
    flows = pd.read_csv(RAW_PATH, usecols=[*model.features, "Label"], dtype=dtype)
    partition = pd.read_csv(PARTITION_PATH, usecols=["benchmark_partition"])
    validation = flows[partition["benchmark_partition"].eq("frozen_validation").to_numpy()]
    matrix = validation.loc[:, list(model.features)].to_numpy(dtype=np.float32)

    alert = calibrated_risk(model, matrix) >= model.thresholds["strict_fpr_0.001"]
    label = validation["Label"].to_numpy() == 1
    confusion = (
        int((~alert & ~label).sum()),
        int((alert & ~label).sum()),
        int((~alert & label).sum()),
        int((alert & label).sum()),
    )
    assert confusion == (458593, 452, 1776, 17234)
