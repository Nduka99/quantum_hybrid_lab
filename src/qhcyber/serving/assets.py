"""Integrity manifest for deployment assets: SHA-256, size and provenance of every file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qhcyber.serving.triage import sha256_bytes

MANIFEST_FILE = "MANIFEST.json"


def read_manifest(asset_dir: Path) -> dict[str, Any]:
    path = asset_dir / MANIFEST_FILE
    if not path.exists():
        return {"schema_version": 1, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(asset_dir: Path, sources: dict[str, dict[str, str]]) -> None:
    """Hash the named files and merge them, with their provenance, into the manifest."""

    manifest = read_manifest(asset_dir)
    for name, source in sources.items():
        content = (asset_dir / name).read_bytes()
        manifest["files"][name] = {"sha256": sha256_bytes(content), "bytes": len(content), **source}
    manifest["files"] = dict(sorted(manifest["files"].items()))
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (asset_dir / MANIFEST_FILE).write_bytes(text.encode("utf-8"))


def verify_manifest(asset_dir: Path) -> dict[str, Any]:
    """Raise if any listed file is missing or altered; return the manifest."""

    manifest = read_manifest(asset_dir)
    if not manifest["files"]:
        raise ValueError(f"No asset manifest in {asset_dir}.")
    for name, entry in manifest["files"].items():
        path = asset_dir / name
        if not path.exists():
            raise ValueError(f"Asset {name} is missing.")
        if sha256_bytes(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"Asset {name} does not match its manifest hash.")
    return manifest
