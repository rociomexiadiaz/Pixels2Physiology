"""Draw a clean ink-only ECG from a time series - the training target for UNet-2."""
import cv2
import numpy as np

from .config import CANVAS_H, CANVAS_W, PX_PER_MV, PX_PER_SEC, STRIP_LEADS

LEAD_IDX = {"I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
            "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11}


def render_ecg(signals, fs, key, row_y=(580, 862, 1150, 1405), col_x=(125, 618, 1108, 1600),
               seconds_per_panel=2.5, panel_h=250, thickness=3):
    """signals: (n, 12) array in LEAD_IDX order. key: strip 0-3. Returns uint8 image."""
    signals = np.asarray(signals, dtype=float)
    fs = float(fs)
    img = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)

    def extract(sig, n_required):
        sig = sig[~np.isnan(sig)]
        if len(sig) < 5:
            return np.zeros(n_required)
        return sig[:n_required] if len(sig) >= n_required else np.concatenate([sig, np.zeros(n_required - len(sig))])

    def draw(sig, x0, y0):
        x = np.round(np.arange(len(sig)) / fs * PX_PER_SEC).astype(int) + x0
        y = (y0 + panel_h // 2 - sig * PX_PER_MV).astype(int)
        ok = (x >= 0) & (x < CANVAS_W)
        x, y = x[ok], y[ok]
        for i in range(len(x) - 1):
            cv2.line(img, (x[i], y[i]), (x[i + 1], y[i + 1]), color=255, thickness=thickness)

    if key < 3:
        n_panel = int(seconds_per_panel * fs)
        for c, lead in enumerate(STRIP_LEADS[key]):
            draw(extract(signals[:, LEAD_IDX[lead]], n_panel), col_x[c], row_y[key])
    else:
        draw(extract(signals[:, LEAD_IDX["II"]], signals.shape[0]), col_x[0], row_y[3])
    return img


def render_target(signals, fs, key):
    """Binary, dilated target exactly as used in training."""
    img = render_ecg(signals, fs, key).astype(np.float32)
    img = (img > 127).astype(np.float32)
    return cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=1)
