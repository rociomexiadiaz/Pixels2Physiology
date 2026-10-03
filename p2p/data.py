"""Datasets and the seeded split used in the paper (training only)."""
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import UNET1_MEAN, UNET1_STD
from .orb import corners_to_mask
from .preprocess import downsample, resize_with_padding

DAMAGED = (3, 4, 5, 6, 9, 10, 11, 12)


def paper_split(meta, seed=42):
    """90% of records for UNet-2 (grid), 10% for UNet-1 (mask); each 80/20 train/val.

    Splits are by record id, so no ECG appears in more than one split.
    """
    grid, mask = train_test_split(meta, test_size=0.1, random_state=seed)
    train_grid, val_grid = train_test_split(grid, test_size=0.2, random_state=seed)
    train_mask, val_mask = train_test_split(mask, test_size=0.2, random_state=seed)
    return train_grid, val_grid, train_mask, val_mask


class CropDataset(Dataset):
    """UNet-1: damaged page -> paper mask. `corners` from scripts/make_masks.py."""

    def __init__(self, metadata, corners: pd.DataFrame, root):
        self.root = root
        self.corners = corners.set_index(["id", "suffix"])
        ids = set(metadata["id"].astype(str))
        self.samples = [k for k in self.corners.index if str(k[0]) in ids]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        id_str, suffix = self.samples[idx]
        img = cv2.imread(f"{self.root}/{id_str}/{id_str}-{suffix:04d}.png", cv2.IMREAD_GRAYSCALE)
        small = downsample(resize_with_padding(img)).astype(np.float32) / 255.0
        small = (small - UNET1_MEAN / 255.0) / (UNET1_STD / 255.0)
        c = self.corners.loc[(id_str, suffix)]
        pts = np.array([[c.x0, c.y0], [c.x1, c.y1], [c.x2, c.y2], [c.x3, c.y3]], dtype=np.float32)
        mask = downsample(corners_to_mask(pts))
        return {"img": torch.from_numpy(small)[None], "mask": torch.from_numpy(mask)[None].float(),
                "suffix": suffix}


class ECGDataset(Dataset):
    """UNet-2: one damaged page per record per epoch (stained types oversampled)."""

    TRAIN_SUFFIXES = (3, 4, 5, 6, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12)

    def __init__(self, metadata, root, suffix_map=None):
        self.meta = metadata.reset_index(drop=True)
        self.root = root
        self.suffix_map = suffix_map

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        r = self.meta.iloc[idx]
        id_str = str(r["id"])
        suffix = self.suffix_map[id_str] if self.suffix_map else int(np.random.choice(self.TRAIN_SUFFIXES))
        img = cv2.imread(f"{self.root}/{id_str}/{id_str}-{suffix:04d}.png", cv2.IMREAD_GRAYSCALE)
        return {"id": id_str, "fs": float(r["fs"]), "lq_image": resize_with_padding(img)}
