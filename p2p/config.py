"""Every fixed number the pipeline relies on, in one place.

These values come straight from the training notebook. Change them here and
every stage picks the change up.
"""

# Canvas every page is padded/warped to (height, width)
CANVAS_H, CANVAS_W = 1700, 2200

# UNet-1 runs on a downsampled page (height, width); width rounded to /16
SMALL_H, SMALL_W = 512, 656

# Normalisation statistics measured on the training set
UNET1_MEAN, UNET1_STD = 193, 55            # 0-255 grayscale
UNET2_MEAN, UNET2_STD = 196 / 255, 42 / 255

# Vertical pixel ranges of the four printed strips on the 1700x2200 canvas.
# Strips 1-3 hold four 2.5 s leads each, strip 4 is the 10 s rhythm lead (II).
STRIP_ROWS = [(510, 894), (797, 1181), (1076, 1460), (1316, 1700)]
STRIP_COL_MARGIN = 4                       # trim 4 px each side -> width 2192

STRIP_LEADS = {
    0: ["I", "aVR", "V1", "V4"],
    1: ["II", "aVL", "V2", "V5"],
    2: ["III", "aVF", "V3", "V6"],
    3: ["II"],
}
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Standard ECG paper: 0.5 mV per 39 px vertically at this resolution
MV_PER_PX = 0.5 / 39

# Rendering scale used to draw training targets
PX_PER_MV = 78
PX_PER_SEC = 196

# Orientation check: a real strip has trace pixels this far from its baseline
ORIENTATION_MARGIN = 50

# Degradation types in the PhysioNet 2025 image set (file suffix -> name)
DEGRADATIONS = {
    1: "Clean print",
    3: "Colour scan",
    4: "Black & white scan",
    5: "Phone photo of a printout",
    6: "Phone photo of a screen",
    9: "Stained photo, sideways",
    10: "Damaged printout",
    11: "Mould, colour scan",
    12: "Mould, black & white scan",
}

WEIGHTS = {
    "unet1": "unet1_best.pth",
    "unet2": "unet2.pth",
}
