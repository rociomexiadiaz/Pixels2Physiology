"""Train UNet-2 (find the ink). Paper settings: Adam 1e-4, 40 epochs, batch 2, Dice+BCE.

Uses a trained UNet-1 to flatten each page first, so UNet-2 learns on exactly
what it will see in production. Targets are rendered from the true signal.

    python scripts/train_unet2.py --data /path/to/physionet-ecg-image-digitization
"""
import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from p2p.crop import crop_document                 # noqa: E402
from p2p.data import ECGDataset, paper_split       # noqa: E402
from p2p.models import DiceFocalLoss, UNet, UNet_Res_CBAM   # noqa: E402
from p2p.pipeline import pick_device               # noqa: E402
from p2p.render import render_target               # noqa: E402
from p2p.strips import to_tensor                   # noqa: E402
from p2p.weights import weight_path                # noqa: E402

# Training patches are taller than the inference strips (592 vs 384 px) for context
TRAIN_ROWS = {0: (406, 998), 1: (693, 1285), 2: (972, 1564), 3: (1108, 1700)}

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--out", default="weights/unet2.pth")
args = ap.parse_args()

random.seed(42); np.random.seed(42); torch.manual_seed(42)
root = f"{args.data}/train"
train_grid, val_grid, _, _ = paper_split(pd.read_csv(f"{args.data}/train.csv"))
rng = np.random.default_rng(0)
val_map = {str(i): int(rng.choice([3, 4, 5, 6, 9, 10, 11, 12])) for i in val_grid["id"]}
collate = lambda b: b  # noqa: E731  keep numpy pages as-is
train_dl = DataLoader(ECGDataset(train_grid, root), batch_size=2, shuffle=True, collate_fn=collate)
val_dl = DataLoader(ECGDataset(val_grid, root, val_map), batch_size=2, collate_fn=collate)

device = pick_device()
unet1 = UNet(); unet1.load_state_dict(torch.load(weight_path("unet1"), map_location="cpu")); unet1.to(device).eval()
unet2 = UNet_Res_CBAM().to(device)
opt = torch.optim.Adam(unet2.parameters(), lr=1e-4)
loss_fn = DiceFocalLoss()


def make_batch(items, keys):
    xs, ys = [], []
    for item, key in zip(items, keys):
        doc, _, _ = crop_document(item["lq_image"], unet1, device)
        y0, y1 = TRAIN_ROWS[key]
        xs.append(to_tensor(doc[y0:y1, 4:-4]))
        signals = pd.read_csv(f"{root}/{item['id']}/{item['id']}.csv").values
        ys.append(torch.from_numpy(render_target(signals, item["fs"], key)[y0:y1, 4:-4])[None])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


best, best_wts = float("inf"), None
for epoch in range(args.epochs):
    unet2.train(); tl = 0
    for items in tqdm(train_dl, desc=f"epoch {epoch + 1} train"):
        x, y = make_batch(items, [random.choice(list(TRAIN_ROWS)) for _ in items])
        opt.zero_grad(); loss = loss_fn(unet2(x), y); loss.backward(); opt.step(); tl += loss.item()
    unet2.eval(); vl = 0; k = 0
    with torch.no_grad():
        for items in val_dl:
            keys = [(k + j) % 4 for j in range(len(items))]; k += len(items)
            x, y = make_batch(items, keys)
            vl += loss_fn(unet2(x), y).item()
    tl, vl = tl / len(train_dl), vl / len(val_dl)
    print(f"epoch {epoch + 1}: train {tl:.4f}  val {vl:.4f}")
    if vl < best:
        best, best_wts = vl, copy.deepcopy(unet2.state_dict())
torch.save(best_wts, args.out)
print("saved", args.out)
