"""Shared pytest setup.

Make the repository root importable so that ``from src.<module> import ...``
works regardless of the directory pytest is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
