"""Run the real pipeline on a few held-out images and save every intermediate
step for the interactive page (docs/) and the README GIF.

    python scripts/make_demo_assets.py --data /path/to/physionet-ecg-image-digitization \
        --samples 987816877:3 3117857513:10 ...

Nothing on the page is drawn by hand: every mask, crop and trace is what the
models produced, and every error number is computed against the true signal.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from p2p import Pipeline                                   # noqa: E402
from p2p.config import CANVAS_H, CANVAS_W, DEGRADATIONS, MV_PER_PX, STRIP_LEADS, STRIP_ROWS   # noqa: E402
from p2p.digitise import clean_mask, extract_ecg_trace, strip_to_leads        # noqa: E402
from p2p.metrics import aligned_scores                     # noqa: E402

INK = (75, 49, 224)          # BGR of #e0314b, the page accent
PLOT_HZ = 100                # signals are thinned to 100 Hz for the browser


def content_box(raw_shape):
    """Where the real image sits inside the padded 1700x2200 canvas."""
    h, w = raw_shape[:2]
    s = min(CANVAS_W / w, CANVAS_H / h)
    nw, nh = int(w * s), int(h * s)
    x0, y0 = (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2
    return x0, y0, nw, nh


def rgba_ink(mask, color=INK, alpha=255):
    m = (mask > 0).astype(np.uint8)
    out = np.zeros((*m.shape, 4), np.uint8)
    out[m == 1] = (*color, alpha)
    return out


def jpg(path, img, width, q=82):
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, q])


def png(path, img, width):
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def pair_in_ink_frame(pred, gt, fs):
    """Recovered trace exactly where it was read off the page, with the original
    recording shifted onto it by the metric's own lag and offset.

    Drawing it this way round keeps the red line on top of the ink it came
    from, so the page can morph from ink to signal; the error is unchanged.
    """
    gt = np.asarray(gt, float)
    gt = gt[~np.isnan(gt)]
    s = aligned_scores(pred, gt, fs)
    lag = int(round(s["lag_s"] * fs))
    L = min(len(pred), len(gt))
    i = np.arange(len(pred))
    k = i + lag
    ok = (i < L) & (k >= 0) & (k < L)
    true_disp = np.full(len(pred), np.nan)
    true_disp[ok] = gt[k[ok]] - s["offset_mv"]
    step = max(1, int(round(fs / PLOT_HZ)))
    return np.asarray(pred)[::step], true_disp[::step], s


def rounded(a):
    return [None if np.isnan(v) else round(float(v), 3) for v in a]


def export_sample(pipe, data, rid, suffix, meta, out_dir):
    raw = cv2.imread(str(data / "train" / rid / f"{rid}-{suffix:04d}.png"))
    fs, n = int(meta.loc[rid, "fs"]), int(meta.loc[rid, "sig_len"])
    t0 = time.perf_counter()
    res = pipe.run(raw)
    runtime = time.perf_counter() - t0
    truth = pd.read_csv(data / "train" / rid / f"{rid}.csv")

    d = out_dir / f"{rid}-{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    x0, y0, nw, nh = content_box(raw.shape)

    # 1. the photo as received
    shown = cv2.resize(raw, (nw, nh), interpolation=cv2.INTER_AREA)
    jpg(d / "input.jpg", shown, 1000)

    # 2. UNet-1's paper mask + fitted corners, in the orientation it was found
    mask = res.doc_mask
    if res.flipped:
        mask = cv2.rotate(mask, cv2.ROTATE_180)   # show on the upright-as-received photo
    m = mask[y0:y0 + nh, x0:x0 + nw]
    png(d / "mask.png", rgba_ink(m, color=(255, 200, 40), alpha=110), 1000)
    corners = None
    if res.corners is not None:
        c = res.corners.copy()
        if res.flipped:
            c = np.array([[CANVAS_W - 1 - x, CANVAS_H - 1 - y] for x, y in c])
        corners = [[round(float((x - x0) / nw), 4), round(float((y - y0) / nh), 4)] for x, y in c]

    # 3. flattened, straightened page
    page = cv2.cvtColor(res.document, cv2.COLOR_GRAY2BGR)
    jpg(d / "page.jpg", page, 1100)

    # 4-5. strips and the ink UNet-2 found in each
    for k, (s_img, s_mask) in enumerate(zip(res.strip_images, res.strip_masks)):
        jpg(d / f"strip{k}.jpg", cv2.cvtColor(s_img, cv2.COLOR_GRAY2BGR), 1096)
        png(d / f"ink{k}.png", rgba_ink(clean_mask(s_mask)), 1096)

    # 6. signals vs truth
    q = n // 4
    leads = {}
    for i, sm in enumerate(res.strip_masks):
        for j, (name, v) in enumerate(strip_to_leads(clean_mask(sm), i, n).items()):
            if name == "II" and i != 3:
                continue
            gt = truth[name].values if i == 3 else truth[name].values[j * q:(j + 1) * q]
            p, t, s = pair_in_ink_frame(v, gt, fs)
            leads[name] = {"strip": i, "col": j, "pred": rounded(p), "true": rounded(t),
                           "rmse": round(s["rmse"], 4), "corr": round(s["corr"], 3)}
    rmses = [v["rmse"] for v in leads.values()]

    # where each strip's ink sits, so the page can place it on the signal axes
    strip_geom = []
    for sm in res.strip_masks:
        cm = clean_mask(sm)
        _, _, base = extract_ecg_trace(cm)
        cols = np.where(cm.any(axis=0))[0]
        strip_geom.append({"w": int(cm.shape[1]), "h": int(cm.shape[0]), "baseline": int(base),
                           "first": int(cols[0]) if len(cols) else 0, "last": int(cols[-1]) if len(cols) else int(cm.shape[1] - 1)})

    info = {
        "id": f"{rid}-{suffix}", "record": rid, "suffix": suffix, "degradation": DEGRADATIONS[suffix],
        "flipped": bool(res.flipped), "aspect": round(nw / nh, 4), "corners": corners,
        "strip_rows": [[a / CANVAS_H, b / CANVAS_H] for a, b in STRIP_ROWS],
        "strip_geom": strip_geom, "mv_per_px": MV_PER_PX,
        "fs_plot": fs / max(1, int(round(fs / PLOT_HZ))), "leads": leads,
        "mean_rmse": round(float(np.mean(rmses)), 4) if rmses else None,
        "n_leads": len(leads),
        "runtime_s": round(runtime, 2),
    }
    (d / "result.json").write_text(json.dumps(info, separators=(",", ":")))
    return info, res, raw


def gif_frames(info, res, raw, width=900):
    """Six-step story of one image, as PIL frames."""
    from PIL import Image, ImageDraw, ImageFont
    H = int(width * CANVAS_H / CANVAS_W)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30)
    except OSError:
        font = ImageFont.load_default()

    def canvas(bgr, title):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = min(width / w, (H - 60) / h)
        im = Image.fromarray(cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA))
        bg = Image.new("RGB", (width, H), (250, 248, 245))
        bg.paste(im, ((width - im.width) // 2, 60 + (H - 60 - im.height) // 2))
        dr = ImageDraw.Draw(bg)
        dr.text((20, 14), title, fill=(30, 30, 40), font=font)
        return bg

    x0, y0, nw, nh = content_box(raw.shape)
    shown = cv2.resize(raw, (nw, nh), interpolation=cv2.INTER_AREA)
    frames = [canvas(shown, "1  A paper ECG, photographed")]

    over = shown.copy()
    mask = res.doc_mask if not res.flipped else cv2.rotate(res.doc_mask, cv2.ROTATE_180)
    m = mask[y0:y0 + nh, x0:x0 + nw] > 0
    over[m] = (0.6 * over[m] + 0.4 * np.array([40, 200, 255])).astype(np.uint8)
    if info["corners"]:
        pts = np.array([[x * nw, y * nh] for x, y in info["corners"]], np.int32)
        cv2.polylines(over, [pts], True, (60, 60, 230), max(3, nw // 300))
    frames.append(canvas(over, "2  AI finds the paper"))

    frames.append(canvas(cv2.cvtColor(res.document, cv2.COLOR_GRAY2BGR),
                         "3  Flatten" + (" + rotate 180°" if res.flipped else " and straighten")))

    boxed = cv2.cvtColor(res.document, cv2.COLOR_GRAY2BGR)
    for a, b in STRIP_ROWS:
        cv2.rectangle(boxed, (4, a), (CANVAS_W - 5, b), (60, 60, 230), 6)
    frames.append(canvas(boxed, "4  Cut the four printed strips"))

    ink = np.vstack([(clean_mask(sm) * 255).astype(np.uint8) for sm in res.strip_masks])
    ink_bgr = np.full((*ink.shape, 3), 250, np.uint8)
    ink_bgr[ink > 0] = INK
    frames.append(canvas(ink_bgr, "5  AI lifts the ink off the grid"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width / 100, (H - 60) / 100), dpi=100)
    lead = info["leads"].get("II")
    if lead:
        tv = np.array(lead["true"], dtype=float)
        t = np.arange(len(tv)) / info["fs_plot"]
        ax.plot(t, tv, color="#2e6fb5", lw=1.4, label="Original recording")
        ax.plot(t, np.array(lead["pred"], dtype=float), color="#e0314b", lw=1.2, label="Recovered from the photo")
        ax.set_title(f"Lead II, error {lead['rmse']:.2f} mV", fontsize=14)
    ax.set_xlabel("seconds"); ax.set_ylabel("mV"); ax.legend(loc="upper right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.canvas.draw()
    plot = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    frames.append(canvas(cv2.cvtColor(plot, cv2.COLOR_RGB2BGR), "6  Back to a digital signal"))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--samples", nargs="+", required=True, help='record:suffix or "record:suffix=Dropdown label"')
    ap.add_argument("--gif", help="record:suffix to turn into docs/assets/pipeline.gif")
    ap.add_argument("--out", default="docs/assets/samples")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    meta = pd.read_csv(data / "train.csv", dtype={"id": str}).set_index("id")
    pipe = Pipeline.from_pretrained()

    index = []
    for item in args.samples:
        key, _, label = item.partition("=")
        rid, suffix = key.split(":")
        info, res, raw = export_sample(pipe, data, rid, int(suffix), meta, out)
        info["label"] = label or info["degradation"]
        (out / info["id"] / "result.json").write_text(json.dumps(info, separators=(",", ":")))
        index.append({k: info[k] for k in ("id", "label", "degradation", "flipped", "mean_rmse", "n_leads", "runtime_s")})
        print(f"{info['id']:>16}  {info['degradation']:<28} rmse={info['mean_rmse']}  flipped={info['flipped']}")
        if args.gif == key:
            frames = gif_frames(info, res, raw)
            frames[0].save(out.parent / "pipeline.gif", save_all=True, append_images=frames[1:],
                           duration=[1600, 1600, 1600, 1600, 1600, 3200], loop=0, optimize=True)
            print("wrote pipeline.gif")
    (out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
