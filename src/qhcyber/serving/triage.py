"""Score NetFlow v2 flows with the frozen LightGBM triage model and its decision policy.

The calibration reproduces ``scripts/build_15_uq_netflow_v2_forensic_errors_and_xai.py``:
risk = expit(intercept + coefficient * logit(clip(raw probability))).
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.special import expit

MODEL_FILE = "lightgbm_behaviour_only.txt"
POLICY_FILE = "lightgbm_behaviour_only_policy.json"
MAX_FLOWS_PER_REQUEST = 100

# Public policy names mapped to the threshold keys frozen in the policy file.
POLICY_THRESHOLD_KEYS = {
    "strict_fpr_0.001": "maximum_recall_under_fpr_0.001",
    "max_macro_f1": "maximum_macro_f1",
    "default_0.5": "default_0.5",
}


@dataclass(frozen=True)
class TriageModel:
    booster: lgb.Booster
    features: tuple[str, ...]
    coefficient: float
    intercept: float
    clip: tuple[float, float]
    thresholds: dict[str, float]
    model_version: str
    # LightGBM prediction is serialised so concurrent requests cannot interleave native calls.
    lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_triage_model(asset_dir: Path) -> TriageModel:
    """Load the booster and policy, and check they describe the same feature contract."""

    model_bytes = (asset_dir / MODEL_FILE).read_bytes()
    policy = json.loads((asset_dir / POLICY_FILE).read_text(encoding="utf-8"))
    booster = lgb.Booster(model_str=model_bytes.decode("utf-8"))
    features = tuple(policy["features_in_order"])
    if tuple(booster.feature_name()) != features:
        raise ValueError("Model features do not match the policy feature contract.")

    calibration = policy["sigmoid_calibration"]
    low, high = calibration["clip"]
    thresholds = {
        name: float(policy["thresholds"][key]) for name, key in POLICY_THRESHOLD_KEYS.items()
    }
    return TriageModel(
        booster=booster,
        features=features,
        coefficient=float(calibration["coefficient"]),
        intercept=float(calibration["intercept"]),
        clip=(float(low), float(high)),
        thresholds=thresholds,
        model_version=sha256_bytes(model_bytes)[:12],
    )


def flows_to_matrix(flows: list[dict[str, float]], features: tuple[str, ...]) -> np.ndarray:
    """Validate flows against the feature contract and return the float32 training layout."""

    if not 1 <= len(flows) <= MAX_FLOWS_PER_REQUEST:
        raise ValueError(f"Send between 1 and {MAX_FLOWS_PER_REQUEST} flows.")
    expected = set(features)
    for position, flow in enumerate(flows):
        missing = sorted(expected - set(flow))
        extra = sorted(set(flow) - expected)
        if missing or extra:
            raise ValueError(f"Flow {position}: missing {missing}, unexpected {extra}.")

    values = np.array([[flow[name] for name in features] for flow in flows], dtype=np.float64)
    if not np.isfinite(values).all() or (values > np.finfo(np.float32).max).any():
        raise ValueError("Feature values must be finite and fit in float32.")
    if (values < 0).any():
        raise ValueError("Feature values must be non-negative raw NetFlow units.")
    # The booster was trained on float32 inputs; casting keeps threshold decisions identical.
    return values.astype(np.float32)


def calibrated_risk(model: TriageModel, matrix: np.ndarray) -> np.ndarray:
    with model.lock:
        raw = np.asarray(model.booster.predict(matrix, num_threads=1), dtype=np.float64)
    clipped = np.clip(raw, *model.clip)
    return expit(model.intercept + model.coefficient * np.log(clipped / (1 - clipped)))


def score_flows(model: TriageModel, flows: list[dict[str, float]], policy: str) -> dict[str, Any]:
    """Return calibrated risk and the alert decision for each flow under a named policy."""

    if policy not in model.thresholds:
        raise ValueError(f"Unknown policy {policy!r}; choose one of {sorted(model.thresholds)}.")
    threshold = model.thresholds[policy]
    risk = calibrated_risk(model, flows_to_matrix(flows, model.features))
    return {
        "model_version": model.model_version,
        "policy": policy,
        "threshold": threshold,
        "results": [{"risk": float(value), "alert": bool(value >= threshold)} for value in risk],
    }
