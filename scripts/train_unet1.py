"""Train UNet-1 (find the paper). Paper settings: AdamW 1e-3, 10 epochs, batch 8, BCE+Dice.

    python scripts/train_unet1.py --data /path/to/physionet-ecg-image-digitization
"""
import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from p2p.data import CropDataset, paper_split   # noqa: E402
from p2p.models import BCEDiceLoss, UNet         # noqa: E402
from p2p.pipeline import pick_device             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--corners", default="results/page_corners.csv")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--out", default="weights/unet1_best.pth")
args = ap.parse_args()

torch.manual_seed(42)
root = f"{args.data}/train"
corners = pd.read_csv(args.corners, dtype={"id": str})
_, _, train_mask, val_mask = paper_split(pd.read_csv(f"{args.data}/train.csv"))
train_ds, val_ds = CropDataset(train_mask, corners, root), CropDataset(val_mask, corners, root)

# Photos with background (suffixes 5, 6, 9, 10) are the hard cases: show them 3x as often
weights = [3.0 if s in {5, 6, 9, 10} else 1.0 for _, s in train_ds.samples]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(42))
train_dl = DataLoader(train_ds, batch_size=8, sampler=sampler)
val_dl = DataLoader(val_ds, batch_size=8)

device = pick_device()
model = UNet().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = BCEDiceLoss()
best, best_wts = float("inf"), None
for epoch in range(args.epochs):
    model.train(); tl = 0
    for b in tqdm(train_dl, desc=f"epoch {epoch + 1} train"):
        opt.zero_grad()
        loss = loss_fn(model(b["img"].to(device)), b["mask"].to(device))
        loss.backward(); opt.step(); tl += loss.item()
    model.eval(); vl = 0
    with torch.no_grad():
        for b in val_dl:
            vl += loss_fn(model(b["img"].to(device)), b["mask"].to(device)).item()
    tl, vl = tl / len(train_dl), vl / len(val_dl)
    print(f"epoch {epoch + 1}: train {tl:.4f}  val {vl:.4f}")
    if vl < best:
        best, best_wts = vl, copy.deepcopy(model.state_dict())
torch.save(best_wts, args.out)
print("saved", args.out)
