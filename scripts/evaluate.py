"""Convenience evaluation entrypoint.

Runs validation evaluation for the residual HGB model against persistence.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate_residual_hgb import main


if __name__ == "__main__":
    main()
