"""Stage 4 - turn a binary ink mask into millivolts. Pure signal processing.

For every pixel column, take the ink pixel furthest from the baseline (keeps
the peaks), convert rows to mV with the paper scale, and resample to the
record's true length with a cubic spline.
"""
import cv2
import numpy as np
from scipy.interpolate import interp1d

from .config import MV_PER_PX, STRIP_LEADS


def clean_mask(mask):
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def extract_ecg_trace(mask):
    """Returns (row per column, mV per column, baseline row). NaN = no ink."""
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape

    rows, _ = np.nonzero(binary)
    baseline_row = int(np.median(rows)) if len(rows) else height // 2

    cols_with_data = np.where(binary.any(axis=0))[0]
    if len(cols_with_data) == 0:
        nan = np.full(width, np.nan)
        return nan, nan.copy(), baseline_row
    first_col, last_col = cols_with_data[0], cols_with_data[-1]

    trace_rows = np.full(width, np.nan)
    for col in range(width):
        ink = np.where(binary[:, col] > 0)[0]
        if len(ink):
            trace_rows[col] = ink[np.argmax(np.abs(ink - baseline_row))]
        elif first_col < col < last_col:
            trace_rows[col] = baseline_row

    voltages = (baseline_row - trace_rows) * MV_PER_PX
    return trace_rows, voltages, baseline_row


def resample_to_length(v, target_len):
    v = np.asarray(v)
    f = interp1d(np.linspace(0, 1, len(v)), v, kind="cubic", fill_value="extrapolate")
    return f(np.linspace(0, 1, target_len))


def strip_to_leads(strip_mask, strip_idx, n_samples):
    """One strip mask -> {lead name: mV array}.

    Strips 1-3 span 10 s split into four 2.5 s leads; strip 4 is 10 s of lead II.
    Returns {} if the strip has no ink at all.
    """
    _, volts, _ = extract_ecg_trace(strip_mask)
    volts = volts[~np.isnan(volts)]
    if len(volts) < 4:
        return {}
    v = resample_to_length(volts, n_samples)

    names = STRIP_LEADS[strip_idx]
    if strip_idx == 3:
        return {"II": v}
    q = n_samples // 4
    return {name: v[j * q:(j + 1) * q] for j, name in enumerate(names)}
