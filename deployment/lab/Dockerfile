# QHybrid Lab image. Build context is the repository root (see compose.yaml).
# Base pinned by digest: python:3.11.17-slim-bookworm, checked 2026-10-08.
FROM python:3.11.17-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=2

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements/lab.lock requirements/lab.lock
RUN pip install --require-hashes --no-deps -r requirements/lab.lock

# The package is used from source (PYTHONPATH) so no unpinned build backend is downloaded.
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY deployment/lab/ deployment/lab/

USER 1000:1000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uvicorn", "app:app", "--app-dir", "deployment/lab", "--host", "0.0.0.0", "--port", "8000", \
     "--limit-concurrency", "32", "--no-server-header"]


FROM runtime AS test
USER root
COPY requirements/lab-test.lock requirements/lab-test.lock
RUN pip install --require-hashes --no-deps -r requirements/lab-test.lock
COPY tests/lab/ tests/lab/
USER 1000:1000
CMD ["python", "-m", "pytest", "tests/lab", "-q", "-p", "no:cacheprovider"]
