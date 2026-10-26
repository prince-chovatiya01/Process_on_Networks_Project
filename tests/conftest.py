"""
Pytest fixtures and configuration.
"""

import sys
from pathlib import Path
import pytest

# Ensure src is on sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


@pytest.fixture
def project_root() -> Path:
    return root_dir
