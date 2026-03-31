#!/usr/bin/env python3
"""Backward-compatible entrypoint for the IE582 food insecurity pipeline.

Preferred invocation (new):
  python -m ie582_food_insecurity.cli

This wrapper keeps previous command usage working:
  python build_food_insecurity_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    from ie582_food_insecurity.cli import main
except Exception as exc:  # pragma: no cover - runtime env specific
    raise SystemExit(
        "Failed to import pipeline package. Ensure dependencies are installed and "
        "run from IE582Project root."
    ) from exc


if __name__ == "__main__":
    main()
