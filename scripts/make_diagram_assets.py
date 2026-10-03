"""Real intermediate outputs for the "under the hood" diagram on docs/index.html.

    python scripts/make_diagram_assets.py --data /path/to/physionet-ecg-image-digitization

Writes docs/assets/diagram/*.jpg and docs/assets/diagram.js:
  orb.jpg       ORB matches between a clean page and a damaged photo + the label polygon
  fft.jpg       the FFT-magnitude channel UNet-2 sees next to the grayscale strip
  orientation   ink-per-row histograms of the same strip read upside down vs upright
  digitise      a zoomed window of an ink mask with the baseline and picked pixels
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from p2p import Pipeline, orient                       # noqa: E402
from p2p.config import MV_PER_PX                       # noqa: E402
from p2p.digitise import clean_mask, extract_ecg_trace  # noqa: E402
from p2p.orb import ORB                                # noqa: E402
from p2p.preprocess import resize_with_padding         # noqa: E402
from p2p.strips import fft_magnitude                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/assets/diagram"


def orb_figure(data, rid, suffix, n_lines=45):
    hq = cv2.imread(str(data / "train" / rid / f"{rid}-0001.png"))
    lq = resize_with_padding(cv2.imread(str(data / "train" / rid / f"{rid}-{suffix:04d}.png")))
    orb = ORB()
    kp1, des1 = orb.keypoint_descriptors(hq)
    kp2, des2 = orb.keypoint_descriptors(lq)
    good = sorted([m for m, n in orb.bf.knnMatch(des1, des2, k=2) if m.distance < 0.7 * n.distance],
                  key=lambda m: m.distance)
    corners = orb.page_corners(hq, lq)

    h = 700
    a = cv2.resize(hq, (int(hq.shape[1] * h / hq.shape[0]), h))
    b = cv2.resize(lq, (int(lq.shape[1] * h / lq.shape[0]), h))
    gap = 60
    canvas = np.full((h, a.shape[1] + gap + b.shape[1], 3), 255, np.uint8)
    canvas[:, :a.shape[1]] = a
    canvas[:, a.shape[1] + gap:] = b
    sa, sb, ox = h / hq.shape[0], h / lq.shape[0], a.shape[1] + gap
    for m in good[:n_lines]:
        p = tuple(int(v * sa) for v in kp1[m.queryIdx].pt)
        q = kp2[m.trainIdx].pt
        q = (int(q[0] * sb) + ox, int(q[1] * sb))
        cv2.line(canvas, p, q, (60, 170, 60), 2, cv2.LINE_AA)
        cv2.circle(canvas, p, 5, (60, 170, 60), -1, cv2.LINE_AA)
        cv2.circle(canvas, q, 5, (60, 170, 60), -1, cv2.LINE_AA)
    if corners is not None:
        pts = np.array([[x * sb + ox, y * sb] for x, y in corners], np.int32)
        cv2.polylines(canvas, [pts], True, (75, 49, 224), 5, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "orb.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return len(good)


def strip_hist(mask):
    m = (mask > 0).astype(np.uint8)
    hist = m.sum(axis=1)
    base = int(np.argmax(hist))
    return {"hist": hist.astype(int).tolist(), "baseline": base, "upright": orient.strip_looks_upright(m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--orb", default="987816877:6")
    ap.add_argument("--flipped", default="1962423499:10")
    ap.add_argument("--fft", default="3201760749:10")
    ap.add_argument("--digitise", default="3002700944:6")
    args = ap.parse_args()
    data = Path(args.data)
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline.from_pretrained()
    out = {}

    rid, s = args.orb.split(":")
    out["orb_matches"] = orb_figure(data, rid, int(s))

    # orientation: same page read upside down (first pass) and upright (kept)
    rid, s = args.flipped.split(":")
    res = pipe.run(data / "train" / rid / f"{rid}-{int(s):04d}.png")
    first = res.first_pass["strip_masks"]
    out["orientation"] = {
        "before": [strip_hist(m) for m in first],
        "after": [strip_hist(m) for m in res.strip_masks],
        "score_before": orient.orientation_score(first)[0],
        "score_after": orient.orientation_score(res.strip_masks)[0],
        "margin": 50,
    }
    for k, m in enumerate(first):
        ink = np.full((*m.shape, 3), 250, np.uint8)
        ink[m > 0] = (75, 49, 224)
        cv2.imwrite(str(OUT / f"flip_before{k}.png"), cv2.resize(ink, (1096, 192), interpolation=cv2.INTER_AREA))

    # FFT channel
    rid, s = args.fft.split(":")
    res = pipe.run(data / "train" / rid / f"{rid}-{int(s):04d}.png")
    strip = res.strip_images[0]
    fft = (fft_magnitude(strip) * 255).astype(np.uint8)
    fft = cv2.equalizeHist(fft)
    cv2.imwrite(str(OUT / "strip_gray.jpg"), cv2.resize(strip, (1096, 192), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
    cv2.imwrite(str(OUT / "strip_fft.jpg"), cv2.resize(fft, (1096, 192), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
    ink = np.full((*strip.shape, 3), 250, np.uint8)
    ink[clean_mask(res.strip_masks[0]) > 0] = (75, 49, 224)
    cv2.imwrite(str(OUT / "strip_ink.png"), cv2.resize(ink, (1096, 192), interpolation=cv2.INTER_AREA))

    # digitise: a window around the tallest peak of row 2
    rid, s = args.digitise.split(":")
    res = pipe.run(data / "train" / rid / f"{rid}-{int(s):04d}.png")
    m = clean_mask(res.strip_masks[1])
    rows, volts, base = extract_ecg_trace(m)
    dev = np.nan_to_num(np.abs(np.asarray(rows, float) - base), nan=0)
    c0 = int(np.clip(np.argmax(dev) - 60, 0, m.shape[1] - 140))
    win = m[:, c0:c0 + 140]
    r0 = int(max(0, min(np.nanmin(rows[c0:c0 + 140]), base) - 12))
    r1 = int(min(m.shape[0], max(np.nanmax(rows[c0:c0 + 140]), base) + 12))
    out["digitise"] = {
        "w": 140, "r0": r0, "r1": r1, "baseline": base - r0,
        "ink": [np.where(win[r0:r1, c] > 0)[0].tolist() for c in range(140)],
        "picked": [None if np.isnan(v) else int(v) - r0 for v in rows[c0:c0 + 140]],
        "mv_per_px": MV_PER_PX,
    }
    (ROOT / "docs/assets/diagram.js").write_text("window.P2P_DIAGRAM = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print({k: v for k, v in out.items() if k in ("orb_matches",)}, "orientation scores",
          out["orientation"]["score_before"], "->", out["orientation"]["score_after"])


if __name__ == "__main__":
    main()
