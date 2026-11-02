"""Alias entry point for find_manifold.py."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.report.find_manifold import main

if __name__ == "__main__":
    main()
