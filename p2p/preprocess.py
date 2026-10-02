"""Stage 0 - make every input the same shape. Pure image processing."""
import cv2
import numpy as np

from .config import CANVAS_H, CANVAS_W, SMALL_H, SMALL_W


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def resize_with_padding(img, target_h=CANVAS_H, target_w=CANVAS_W):
    """Scale to fit the canvas, keeping aspect ratio, and pad the rest with black."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if img.ndim == 2:
        padded = np.zeros((target_h, target_w), dtype=img.dtype)
    else:
        padded = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded


def downsample(img):
    """Shrink the canvas for UNet-1 (512 x 656)."""
    return cv2.resize(img, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA)


def order_corners_clockwise(pts):
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top, bottom = y_sorted[:2], y_sorted[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)
