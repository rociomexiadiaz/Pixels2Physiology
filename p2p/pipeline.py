"""The whole automation, end to end, with every intermediate kept.

    image -> pad -> [UNet-1] paper mask -> flatten page -> cut 4 strips
          -> [UNet-2] ink masks -> upside-down? rotate & redo -> mV signals

Two small neural networks do the two genuinely hard visual jobs (where is the
paper, where is the ink). Everything else is deterministic image and signal
processing, which is cheap, explainable and easy to test.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch

from . import crop, digitise, orient, strips
from .config import LEADS, STRIP_LEADS
from .models import UNet, UNet_Res_CBAM
from .preprocess import load_gray, resize_with_padding
from .weights import weight_path


def pick_device(device=None):
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PipelineResult:
    input_image: np.ndarray            # padded 1700x2200 grayscale
    doc_mask: np.ndarray               # UNet-1 paper mask
    corners: Optional[np.ndarray]      # 4 page corners on the padded image
    document: np.ndarray               # flattened page
    flipped: bool                      # was it rotated 180 degrees?
    strip_images: list                 # 4 grayscale strips
    strip_masks: list                  # 4 binary ink masks
    leads: dict = field(default_factory=dict)  # lead -> mV array (filled by .digitise)
    first_pass: Optional[dict] = None  # what the page looked like before a 180 flip

    def digitise(self, n_samples):
        """Fill .leads for a record of n_samples (10 s * fs).

        Strip 4 (10 s rhythm strip) owns lead II; the 2.5 s copy on strip 2 is
        only used if strip 4 came out empty.
        """
        per_strip = [digitise.strip_to_leads(digitise.clean_mask(m), i, n_samples)
                     for i, m in enumerate(self.strip_masks)]
        self.leads = {}
        for i, found in enumerate(per_strip):
            for name, v in found.items():
                if name == "II" and i != 3:
                    continue
                self.leads[name] = v
        if "II" not in self.leads and "II" in per_strip[1]:
            self.leads["II"] = per_strip[1]["II"]
        return self.leads

    def signals_dataframe(self, fs, n_samples):
        """Challenge-format table: one column per lead, NaN outside each lead's 2.5 s window."""
        if not self.leads:
            self.digitise(n_samples)
        q = n_samples // 4
        out = pd.DataFrame(np.nan, index=range(n_samples), columns=LEADS)
        for name, v in self.leads.items():
            if len(v) == n_samples:
                out[name] = v
            else:
                j = next(names.index(name) for names in STRIP_LEADS.values() if name in names)
                out.iloc[j * q:j * q + len(v), out.columns.get_loc(name)] = v
        out.index.name = "sample"
        return out


class Pipeline:
    def __init__(self, unet1, unet2, device, orientation="compare"):
        self.unet1 = unet1.to(device).eval()
        self.unet2 = unet2.to(device).eval()
        self.device = device
        self.orientation = orientation   # "compare" (default) or "notebook" (paper)

    @classmethod
    def from_pretrained(cls, weights_dir=None, device=None, orientation="compare"):
        device = pick_device(device)
        kw = {"weights_dir": weights_dir} if weights_dir else {}
        u1, u2 = UNet(), UNet_Res_CBAM()
        u1.load_state_dict(torch.load(weight_path("unet1", **kw), map_location="cpu"))
        u2.load_state_dict(torch.load(weight_path("unet2", **kw), map_location="cpu"))
        return cls(u1, u2, device, orientation)

    def _once(self, page):
        document, mask, corners = crop.crop_document(page, self.unet1, self.device)
        document = orient.make_landscape(document)
        strip_imgs = strips.cut_strips(document)
        strip_masks = strips.segment_strips(strip_imgs, self.unet2, self.device)
        return document, mask, corners, strip_imgs, strip_masks

    def run(self, image, n_samples=None):
        """image: path or uint8 array (gray or BGR). n_samples: also digitise."""
        if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
            image = load_gray(image)
        elif image.ndim == 3:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        page = resize_with_padding(image)

        document, mask, corners, s_imgs, s_masks = self._once(page)
        flipped = orient.needs_flip(s_masks)
        first_pass = None
        if flipped:
            first_pass = {"page": page, "document": document, "mask": mask, "corners": corners,
                          "strip_images": s_imgs, "strip_masks": s_masks}
            rot = orient.rotate_180(page)
            second = self._once(rot)
            if self.orientation == "notebook" or orient.orientation_score(second[4]) > orient.orientation_score(s_masks):
                page = rot
                document, mask, corners, s_imgs, s_masks = second
            else:
                flipped = False   # rule fired, but the upright reading was better

        result = PipelineResult(page, mask, corners, document, flipped, s_imgs, s_masks,
                                first_pass=first_pass)
        if n_samples:
            result.digitise(n_samples)
        return result
