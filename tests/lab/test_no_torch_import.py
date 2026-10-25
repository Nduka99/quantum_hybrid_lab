"""The deployment must not pull in torch: the image does not install it."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_service_imports_without_torch() -> None:
    code = (
        "import sys; sys.path.insert(0, 'deployment/lab'); import app; "
        "assert 'torch' not in sys.modules, 'torch was imported'"
    )
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, check=True)
