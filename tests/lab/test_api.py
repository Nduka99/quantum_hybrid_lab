"""HTTP contract of the QHybrid Lab service, including its abuse limits."""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

APP_DIR = Path(__file__).resolve().parents[2] / "deployment" / "lab"
sys.path.insert(0, str(APP_DIR))

import app as service  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(service.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def example_request() -> dict:
    return json.loads((APP_DIR / "assets" / "example_request.json").read_text(encoding="utf-8"))


def test_health_reports_verified_assets(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["verified_assets"] >= 10


def test_page_is_served_with_security_headers(client) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "QHybrid Lab" in page.text
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert "content-security-policy" not in client.get("/docs").headers


def test_scores_example_flows(client, example_request) -> None:
    response = client.post("/v1/triage/score", json=example_request)
    assert response.status_code == 200
    assert len(response.json()["results"]) == len(example_request["flows"])


def test_rejects_invalid_triage_requests(client, example_request) -> None:
    flow = dict(example_request["flows"][0])
    flow.pop("IN_BYTES")
    assert client.post("/v1/triage/score", json={"flows": [flow]}).status_code == 422
    bad_policy = {**example_request, "policy": "most_alerts"}
    assert client.post("/v1/triage/score", json=bad_policy).status_code == 422
    too_many = {"flows": [example_request["flows"][0]] * 101}
    assert client.post("/v1/triage/score", json=too_many).status_code == 422


def test_accepts_the_largest_allowed_batch(client, example_request) -> None:
    batch = {"flows": [example_request["flows"][0]] * service.MAX_FLOWS_PER_REQUEST}
    assert client.post("/v1/triage/score", json=batch).status_code == 200


def test_rejects_oversized_and_unsized_bodies(client) -> None:
    oversized = b"x" * (service.MAX_BODY_BYTES + 1)
    headers = {"content-type": "application/json"}
    assert client.post("/v1/triage/score", content=oversized, headers=headers).status_code == 413
    chunked = client.post("/v1/triage/score", content=iter([b"{}"]), headers=headers)
    assert chunked.status_code == 411


def test_lab_run_returns_a_matched_report(client) -> None:
    request = {"dataset": "breast_cancer", "n_qubits": 2, "reps": 1, "n_train": 40, "seed": 1}
    report = client.post("/v1/lab/run", json=request).json()
    assert report["config"]["n_qubits"] == 2
    assert {model["tier"] for model in report["models"]} >= {"matched quantum"}


def test_lab_run_limits(client) -> None:
    too_wide = {"dataset": "breast_cancer", "n_qubits": 9}
    assert client.post("/v1/lab/run", json=too_wide).status_code == 422
    assert service.LAB_SLOT.acquire(blocking=False)
    try:
        busy = client.post("/v1/lab/run", json={"dataset": "breast_cancer", "n_qubits": 2})
        assert busy.status_code == 429
    finally:
        service.LAB_SLOT.release()


def test_reference_and_dataset_endpoints(client) -> None:
    assert len(client.get("/v1/lab/datasets").json()) == 2
    assert len(client.get("/v1/lab/reference").json()["runs"]) == 80
    assert client.get("/figures/accuracy_by_qubits.png").status_code == 200
