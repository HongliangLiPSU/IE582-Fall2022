#!/usr/bin/env python3
"""Backward-compatible map entrypoint for county index visualization.

Preferred invocation (new):
  PYTHONPATH=src python -m ie582_food_insecurity.map_cli

This wrapper supports:
  python build_food_insecurity_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    from ie582_food_insecurity.map_cli import main
except Exception as exc:  # pragma: no cover - runtime env specific
    raise SystemExit(
        "Failed to import map package. Ensure dependencies are installed and "
        "run from IE582Project root."
    ) from exc


if __name__ == "__main__":
    main()
