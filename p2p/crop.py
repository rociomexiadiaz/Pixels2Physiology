"""Stage 1 - find the paper and flatten it.

Deep learning (UNet-1) predicts which pixels are paper; classical image
processing (morphology, contours, a rotated rectangle, a perspective warp)
turns that mask into a straight, full-canvas page.
"""
import cv2
import numpy as np
import torch

from .config import CANVAS_H, CANVAS_W, UNET1_MEAN, UNET1_STD
from .preprocess import downsample, order_corners_clockwise


@torch.no_grad()
def predict_document_mask(page, model, device):
    """page: 1700x2200 uint8 grayscale -> 1700x2200 {0,1} mask."""
    small = downsample(page).astype(np.float32) / 255.0
    small = (small - UNET1_MEAN / 255.0) / (UNET1_STD / 255.0)
    tensor = torch.from_numpy(small)[None, None].to(device)
    prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    mask_small = (prob > 0.5).astype(np.uint8)
    return cv2.resize(mask_small, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_NEAREST)


def document_corners(mask):
    """Largest blob -> best-fit rotated rectangle -> 4 ordered corners (or None)."""
    mask = (mask > 0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(largest)).astype(np.float32)
    return order_corners_clockwise(box)


def warp_to_canvas(page, corners):
    if corners is None:
        return page
    dst = np.float32([[0, 0], [CANVAS_W - 1, 0], [CANVAS_W - 1, CANVAS_H - 1], [0, CANVAS_H - 1]])
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(page, M, (CANVAS_W, CANVAS_H))


def crop_document(page, model, device):
    mask = predict_document_mask(page, model, device)
    corners = document_corners(mask)
    return warp_to_canvas(page, corners), mask, corners
