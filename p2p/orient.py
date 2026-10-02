"""Stage 2b - is the page upside down? Pure rule-based check.

A correctly oriented ECG strip has ink well above and/or below its busiest
row. If any strip has nothing beyond +/-50 px of that row, the page may be
upside down, so it is rotated 180 degrees and re-run.

mode="notebook" keeps the rotated result whenever the rule fires (as in the
paper). mode="compare" (default) keeps whichever orientation scores better,
because a quiet lead such as aVR can look flat on an upright page and
trigger the rule falsely (31% of validation pages under the notebook rule).
"""
import cv2
import numpy as np

from .config import ORIENTATION_MARGIN


def strip_looks_upright(strip_mask, margin=ORIENTATION_MARGIN):
    bin_img = strip_mask * 255 if strip_mask.max() <= 1 else strip_mask
    hist = np.sum(bin_img == 255, axis=1)
    baseline_row = int(np.argmax(hist))
    above = np.any(bin_img[0:max(baseline_row - margin, 0), :] > 0)
    below = np.any(bin_img[baseline_row + margin:, :] > 0)
    return bool(above or below)


def needs_flip(strip_masks):
    """The notebook rule: any strip that looks flat means 'upside down'."""
    return any(not strip_looks_upright(s) for s in strip_masks)


def orientation_score(strip_masks):
    """(strips that look like real traces, total ink pixels) - higher is better.

    On an upside-down page the fixed strip windows land on margins and
    headers, so fewer strips hold a proper trace and less ink is found.
    """
    return (sum(strip_looks_upright(s) for s in strip_masks), float(sum(s.sum() for s in strip_masks)))


def make_landscape(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if img.shape[0] > img.shape[1] else img


def rotate_180(img):
    return cv2.rotate(img, cv2.ROTATE_180)
