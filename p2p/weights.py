"""Find or fetch the trained weights.

Looks in ./weights first; otherwise downloads from the GitHub release.
"""
import hashlib
import os
import urllib.request
from pathlib import Path

from .config import WEIGHTS

RELEASE_URL = "https://github.com/rociomexiadiaz/Pixels2Physiology/releases/download/v1.0/{name}"
DEFAULT_DIR = Path(os.environ.get("P2P_WEIGHTS", Path(__file__).resolve().parent.parent / "weights"))


def weight_path(key, weights_dir=DEFAULT_DIR, download=True):
    name = WEIGHTS[key]
    path = Path(weights_dir) / name
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = RELEASE_URL.format(name=name)
    print(f"Downloading {name} from {url}")
    urllib.request.urlretrieve(url, path)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
