# QHybrid Lab service

A small web service with two parts:

- **Triage:** scores NetFlow v2 flows with the frozen LightGBM model from the classical benchmark. It returns a
  calibrated risk and an alert decision under one of three frozen policies.
- **Hybrid lab (sandbox):** compares a bandwidth-tuned quantum-kernel SVM with logistic regression,
  linear and RBF SVMs on identical inputs and folds, plus a full-feature LightGBM control. Results
  are a demonstration, not research evidence (see the root README).

All logic lives in `src/qhcyber` (`serving`, `quantum`, `lab`); `app.py` only validates requests.

## Run

```bash
docker compose up --build lab        # first build takes a few minutes
```

Open <http://127.0.0.1:8765>. API documentation: <http://127.0.0.1:8765/docs>.

Without Docker, from the repository root after installing `requirements/lab.lock`:

```bash
PYTHONPATH=src python -m uvicorn app:app --app-dir deployment/lab --port 8765
```

## Test

```bash
docker compose run --rm test                         # tests/lab inside the image
PYTHONPATH=src python -m pytest tests                # same tests, local environment
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, triage model version, number of verified assets |
| GET | `/v1/triage/model-card` | Metrics, shortcut audit, limitations, data citation |
| GET | `/v1/triage/examples` | 20 example flows with their expected risk |
| POST | `/v1/triage/score` | `{"policy": "strict_fpr_0.001", "flows": [{...29 features...}]}` |
| GET | `/v1/lab/datasets` | Lab datasets with field, licence and citation |
| POST | `/v1/lab/run` | `{"dataset": "network_flows", "n_qubits": 4, "reps": 2, "entanglement": "linear", "n_train": 200, "seed": 0}` |
| GET | `/v1/lab/reference` | The pre-registered 80-run reference grid |

## Limits

- Listens on 127.0.0.1 only, in a read-only container with no Linux capabilities.
- The service refuses to start if any file in `assets/` differs from `assets/MANIFEST.json`.
- Request bodies over 128 KB are rejected (413), as are bodies without a length (411).
- At most 100 flows per request. Lab runs allow 2–8 qubits and up to 200 training rows, and only one
  lab run at a time (a second gets 429).
- Nothing a visitor sends is stored or logged beyond the access log line.

## Share it temporarily

To show it to someone remotely, open a Cloudflare Quick Tunnel from the machine running it:

```bash
cloudflared tunnel --url http://127.0.0.1:8765
```

It prints a random `https://….trycloudflare.com` address that works until you stop the command
(Ctrl+C). Quick tunnels have no uptime guarantee and do not support server-sent events; this
service does not use them.

## Rebuild the reference grid

The model, examples and lab sample come from a private research pipeline (see the root README).
The reference grid and its figures can be rebuilt here (needs matplotlib 3.10.8):

```bash
PYTHONPATH=src python scripts/build_lab_reference_runs.py    # 80 runs and figures
```

The run is deterministic and updates `assets/MANIFEST.json`.
