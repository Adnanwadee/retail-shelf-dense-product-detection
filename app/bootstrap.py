"""Runtime bootstrap helpers for the Streamlit demo."""

from __future__ import annotations

import sys
from pathlib import Path


def add_project_src_to_path() -> None:
    """Add the repository src directory to sys.path for local Streamlit execution."""
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"

    if not src_path.exists():
        raise RuntimeError(f"Expected src directory was not found: {src_path}")

    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
