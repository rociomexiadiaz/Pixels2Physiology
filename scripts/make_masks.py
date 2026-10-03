"""Automatically label where the paper is in every damaged image (ORB + RANSAC).

    python scripts/make_masks.py --data /path/to/physionet-ecg-image-digitization
Writes results/page_corners.csv used by train_unet1.py.
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from p2p.data import DAMAGED, paper_split   # noqa: E402
from p2p.orb import ORB                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="results/page_corners.csv")
args = ap.parse_args()

root = Path(args.data) / "train"
_, _, train_mask, val_mask = paper_split(pd.read_csv(Path(args.data) / "train.csv"))
orb = ORB()
records, failed = [], 0
for id_str in tqdm(pd.concat([train_mask, val_mask])["id"].astype(str)):
    hq = cv2.imread(str(root / id_str / f"{id_str}-0001.png"))
    kp1, des1 = orb.keypoint_descriptors(hq)
    for s in DAMAGED:
        lq = cv2.imread(str(root / id_str / f"{id_str}-{s:04d}.png"))
        H = orb.homography(kp1, des1, lq) if lq is not None else None
        if H is None:
            failed += 1
            continue
        corners = orb.page_corners(hq, lq)
        rec = {"id": id_str, "suffix": s}
        for i, (x, y) in enumerate(corners):
            rec[f"x{i}"], rec[f"y{i}"] = float(x), float(y)
        records.append(rec)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(args.out, index=False)
print(f"{len(records)} labelled, {failed} images ORB could not align (dropped)")
