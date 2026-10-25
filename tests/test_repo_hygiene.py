"""Shared files must not leak machine-specific home directories."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Windows user profiles with any separator (including JSON-escaped) and POSIX homes.
HOME_PATH = re.compile(
    r"[A-Za-z]:(?:\\\\|\\|/)Users(?:\\\\|\\|/)[A-Za-z0-9._-]+"
    r"|/(?:home|Users)/[A-Za-z0-9._-]+/"
)


def committable_files() -> list[Path]:
    """Files git would commit: tracked plus untracked files that are not ignored."""

    if shutil.which("git") is None or not (PROJECT_ROOT / ".git").exists():
        pytest.skip("requires a git checkout")
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    )
    names = listing.stdout.decode("utf-8").split("\0")
    return [PROJECT_ROOT / name for name in names if name]


def test_no_home_directory_paths_in_committable_text_files() -> None:
    offenders = []
    for path in committable_files():
        if not path.is_file():
            continue
        content = path.read_bytes()
        if b"\0" in content[:8192]:
            continue  # binary file
        match = HOME_PATH.search(content.decode("utf-8", errors="ignore"))
        if match:
            offenders.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}: {match.group(0)}")
    assert offenders == []
