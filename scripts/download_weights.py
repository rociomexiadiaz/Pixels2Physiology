"""Fetch the two trained models into ./weights (about 250 MB)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from p2p.config import WEIGHTS   # noqa: E402
from p2p.weights import sha256, weight_path   # noqa: E402

for key in WEIGHTS:
    p = weight_path(key)
    print(f"{p}  sha256={sha256(p)[:16]}...")
