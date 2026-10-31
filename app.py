"""Vercel entrypoint.

Vercel looks for a Flask instance named ``app`` in ``app.py``. The real
application lives in ``backend/app.py`` so it can still be run directly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import app  # noqa: E402

__all__ = ["app"]
