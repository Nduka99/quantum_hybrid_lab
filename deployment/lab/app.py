"""QHybrid Lab: a thin HTTP layer over qhcyber.serving (triage) and qhcyber.lab (sandbox).

Run locally:  uvicorn app:app --app-dir deployment/lab --port 8765
All logic lives in src/qhcyber; this file only validates requests and shapes responses.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from qhcyber.lab.benchmark import LabConfig, run_matched_benchmark
from qhcyber.lab.datasets import DATASET_KEYS, dataset_summary, load_dataset
from qhcyber.serving.assets import verify_manifest
from qhcyber.serving.triage import MAX_FLOWS_PER_REQUEST, load_triage_model, score_flows

APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = Path(os.environ.get("QHC_ASSETS_DIR", APP_DIR / "assets"))
MAX_BODY_BYTES = 128 * 1024  # fits MAX_FLOWS_PER_REQUEST flows of 29 features
LAB_TEST_ROWS = 200
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

# Refuse to start on altered or missing assets rather than serve a different model.
MANIFEST = verify_manifest(ASSET_DIR)
TRIAGE_MODEL = load_triage_model(ASSET_DIR)
DATASETS = {key: load_dataset(key, ASSET_DIR) for key in DATASET_KEYS}
# One lab run at a time: a public tunnel must not be able to queue unbounded CPU work.
LAB_SLOT = threading.Semaphore(1)


def read_asset_json(name: str) -> Any:
    return json.loads((ASSET_DIR / name).read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Pay Qiskit's first-transpile cost at start-up instead of on the first visitor's run.
    warm_up = LabConfig("breast_cancer", 2, reps=1, n_train=40, n_test=40)
    await run_in_threadpool(run_matched_benchmark, DATASETS["breast_cancer"], warm_up)
    yield


app = FastAPI(
    title="QHybrid Lab",
    version="0.1.0",
    summary="NetFlow triage and a matched classical/quantum-kernel sandbox.",
    lifespan=lifespan,
)


@app.middleware("http")
async def guard_requests(request: Request, call_next):
    if request.method == "POST":
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return JSONResponse({"detail": "Content-Length is required."}, status_code=411)
        if int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request body is too large."}, status_code=413)
    response = await call_next(request)
    headers = dict(SECURITY_HEADERS)
    if request.url.path.startswith(("/docs", "/redoc")):
        del headers["Content-Security-Policy"]  # Swagger UI loads its scripts from a CDN.
    response.headers.update(headers)
    return response


class ScoreRequest(BaseModel):
    policy: Literal["strict_fpr_0.001", "max_macro_f1", "default_0.5"] = "strict_fpr_0.001"
    flows: list[dict[str, float]] = Field(min_length=1, max_length=MAX_FLOWS_PER_REQUEST)


class LabRequest(BaseModel):
    dataset: Literal["network_flows", "breast_cancer"]
    n_qubits: int = Field(4, ge=2, le=8)
    reps: int = Field(2, ge=1, le=2)
    entanglement: Literal["linear", "full"] = "linear"
    n_train: int = Field(200, ge=40, le=200)
    seed: int = Field(0, ge=0, le=99)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "triage_model_version": TRIAGE_MODEL.model_version,
        "verified_assets": len(MANIFEST["files"]),
    }


@app.get("/v1/triage/model-card")
def model_card() -> Any:
    return read_asset_json("model_card.json")


@app.get("/v1/triage/examples")
def triage_examples() -> Any:
    return read_asset_json("example_flows.json")


@app.post("/v1/triage/score")
def triage_score(request: ScoreRequest) -> dict[str, Any]:
    try:
        return score_flows(TRIAGE_MODEL, request.flows, request.policy)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/v1/lab/datasets")
def lab_datasets() -> list[dict[str, Any]]:
    return [dataset_summary(dataset) for dataset in DATASETS.values()]


@app.get("/v1/lab/reference")
def lab_reference() -> Any:
    return read_asset_json("reference_runs.json")


@app.post("/v1/lab/run")
async def lab_run(request: LabRequest) -> dict[str, Any]:
    if not LAB_SLOT.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="A lab run is in progress; retry shortly.")
    try:
        config = LabConfig(
            dataset=request.dataset,
            n_qubits=request.n_qubits,
            reps=request.reps,
            entanglement=request.entanglement,
            n_train=request.n_train,
            n_test=LAB_TEST_ROWS,
            seed=request.seed,
        )
        return await run_in_threadpool(run_matched_benchmark, DATASETS[request.dataset], config)
    finally:
        LAB_SLOT.release()


app.mount("/figures", StaticFiles(directory=ASSET_DIR / "figures"), name="figures")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html")
