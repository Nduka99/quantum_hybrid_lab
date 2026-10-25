"""Run the pre-registered reference grid for the hybrid lab and save results and figures.

Grid fixed before it was run (README, "Method"): datasets
{network_flows, breast_cancer} x qubits {2, 4, 8, 12} x seeds 0-4 x {tuned bandwidth, fixed pi}.

Run from the repository root (about 10 minutes):
    .quantumstudy/Scripts/python scripts/build_lab_reference_runs.py
Re-draw the figures from saved results only:
    .quantumstudy/Scripts/python scripts/build_lab_reference_runs.py --figures-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qhcyber.lab.benchmark import (  # noqa: E402
    BANDWIDTHS,
    MATCHED_CLASSICAL,
    MATCHED_QUANTUM,
    LabConfig,
    run_matched_benchmark,
)
from qhcyber.lab.datasets import DATASET_KEYS, load_dataset  # noqa: E402
from qhcyber.serving.assets import update_manifest  # noqa: E402

ASSET_DIR = PROJECT_ROOT / "deployment/lab/assets"
RESULTS_FILE = "reference_runs.json"
QUBITS = (2, 4, 8, 12)
SEEDS = (0, 1, 2, 3, 4)
VARIANTS = {"tuned": BANDWIDTHS, "fixed_pi": (float(np.pi),)}


def compact(report: dict, variant: str) -> dict:
    """Keep what the figures and the interface need from one full report."""

    quantum = report["quantum"]
    return {
        "dataset": report["config"]["dataset"],
        "qubits": report["config"]["n_qubits"],
        "seed": report["config"]["seed"],
        "variant": variant,
        "models": {
            model["name"]: {
                "tier": model["tier"],
                "balanced_accuracy": model["test"]["balanced_accuracy"],
                "pr_auc": model["test"]["pr_auc"],
                "hyperparameters": model["hyperparameters"],
            }
            for model in report["models"]
        },
        "selected_bandwidth": quantum["selected_bandwidth"],
        "quantum_kernel": quantum["kernel_diagnostics"],
        "rbf_kernel": quantum["rbf_kernel_diagnostics"],
        "resources": quantum["resources"],
        "elapsed_seconds": report["elapsed_seconds"],
    }


def run_grid() -> list[dict]:
    datasets = {key: load_dataset(key, ASSET_DIR) for key in DATASET_KEYS}
    runs = []
    for key, dataset in datasets.items():
        for qubits in QUBITS:
            for seed in SEEDS:
                for variant, bandwidths in VARIANTS.items():
                    config = LabConfig(key, qubits, seed=seed, bandwidths=bandwidths)
                    runs.append(compact(run_matched_benchmark(dataset, config), variant))
                    print(f"{key:14s} q={qubits:2d} seed={seed} {variant:8s} done", flush=True)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    results_path = ASSET_DIR / RESULTS_FILE
    if not args.figures_only:
        started = time.perf_counter()
        document = {
            "protocol": (
                "Fixed before running (README, Method): 200 train / 200 test rows "
                "per run, 5-fold CV shared by matched models, ZZ feature map with 2 reps and "
                "linear entanglement."
            ),
            "matched_tiers": [MATCHED_CLASSICAL, MATCHED_QUANTUM],
            "runs": run_grid(),
        }
        text = json.dumps(document, indent=1, sort_keys=True) + "\n"
        results_path.write_bytes(text.encode("utf-8"))
        print(f"grid finished in {time.perf_counter() - started:.0f} s")

    from lab_reference_figures import draw_figures  # matplotlib is a build-time dependency only

    document = json.loads(results_path.read_text(encoding="utf-8"))
    figure_names = draw_figures(document["runs"], ASSET_DIR / "figures")
    source = {
        "source": "scripts/build_lab_reference_runs.py",
        "terms": "see the README data notice",
    }
    update_manifest(ASSET_DIR, {RESULTS_FILE: source} | {name: source for name in figure_names})


if __name__ == "__main__":
    main()
