"""Make the scripts under src/ importable from the tests."""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
for sub in ("data", "features", "retrieval", "ranking", "serving", "pipeline"):
    sys.path.insert(0, str(ROOT / "src" / sub))
