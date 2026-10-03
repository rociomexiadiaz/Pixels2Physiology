"""Stages 2-3 - cut the page into its four printed strips and segment the ink.

Strip cropping is fixed geometry (a standard 12-lead layout). Each strip is
fed to UNet-2 with two channels: the grayscale pixels, and the FFT magnitude,
because the printed grid is periodic and shows up as sharp peaks there.
"""
import numpy as np
import torch

from .config import STRIP_COL_MARGIN, STRIP_ROWS, UNET2_MEAN, UNET2_STD


def cut_strips(doc):
    m = STRIP_COL_MARGIN
    return [doc[y0:y1, m:-m] for y0, y1 in STRIP_ROWS]


def fft_magnitude(img):
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log(np.abs(f) + 1)
    return (mag - mag.min()) / (mag.max() - mag.min())


def to_tensor(strip):
    gray = (strip / 255).astype(np.float32)
    t = torch.from_numpy(np.stack([gray, fft_magnitude(strip)], axis=0)).float()
    t[0] = (t[0] - UNET2_MEAN) / UNET2_STD
    return t


@torch.no_grad()
def segment_strips(strips, model, device, batch_size=2):
    """List of strip images -> list of {0,1} float trace masks, same shapes."""
    tensors = [to_tensor(s) for s in strips]
    out = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i:i + batch_size]).to(device)
        probs = torch.sigmoid(model(batch))
        out.extend((probs > 0.5).float().cpu()[:, 0].numpy())
    return out
