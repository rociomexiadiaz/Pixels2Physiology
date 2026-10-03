"""Training-data automation: generate UNet-1's paper masks for free.

Every damaged image has a clean original. ORB keypoints + RANSAC find the
homography between them, and the clean page's corners mapped into the damaged
photo give a perfect "where is the paper" label - no manual annotation.
"""
import cv2
import numpy as np

from .config import CANVAS_H, CANVAS_W
from .preprocess import resize_with_padding


class ORB:
    def __init__(self, match_ratio=0.7, ransac_thresh=5.0, n_features=1000):
        self.match_ratio = match_ratio
        self.ransac_thresh = ransac_thresh
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def keypoint_descriptors(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return self.orb.detectAndCompute(gray, None)

    def homography(self, kp1, des1, lq_image):
        lq_image = resize_with_padding(lq_image)
        kp2, des2 = self.keypoint_descriptors(lq_image)
        if des1 is None or des2 is None:
            return None
        good = [m for m, n in self.bf.knnMatch(des1, des2, k=2) if m.distance < self.match_ratio * n.distance]
        if len(good) < 4:
            return None
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(dst, src, cv2.RANSAC, self.ransac_thresh)
        return H if H is not None and H.shape == (3, 3) else None

    def page_corners(self, hq_image, lq_image):
        """Corners of the clean page inside the (padded) damaged image, or None."""
        kp1, des1 = self.keypoint_descriptors(hq_image)
        H = self.homography(kp1, des1, lq_image)
        if H is None:
            return None
        src = np.float32([[0, 0], [CANVAS_W - 1, 0], [CANVAS_W - 1, CANVAS_H - 1], [0, CANVAS_H - 1]])
        return cv2.perspectiveTransform(src.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(4, 2)


def corners_to_mask(corners):
    mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 1)
    return mask
