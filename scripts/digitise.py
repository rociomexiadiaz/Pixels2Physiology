"""Digitise one ECG image (or a folder of them) to CSV.

    python scripts/digitise.py scan.png --fs 500 --out out/
    python scripts/digitise.py folder_of_scans/ --fs 500 --out out/ --figure

Writes <name>.csv (12 lead columns, millivolts) and optionally <name>.png,
a one-page check sheet a human can eyeball.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from p2p import Pipeline  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def check_sheet(result, path, fs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [3, 2, 2]})
    axes[0].imshow(result.document, cmap="gray"); axes[0].set_title("Flattened page"); axes[0].axis("off")
    axes[1].imshow(np.vstack(result.strip_masks), cmap="gray"); axes[1].set_title("Ink found"); axes[1].axis("off")
    v = result.leads.get("II")
    if v is not None:
        axes[2].plot(np.arange(len(v)) / fs, v, color="#c8102e", lw=0.8)
    axes[2].set_title("Lead II (mV)"); axes[2].set_xlabel("seconds")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("--fs", type=int, default=500, help="sampling rate of the output (Hz)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()

    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT) if src.is_dir() else [src]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    n = int(args.fs * args.seconds)

    pipe = Pipeline.from_pretrained(device=args.device)
    for f in files:
        try:
            r = pipe.run(f, n_samples=n)
            r.signals_dataframe(args.fs, n).to_csv(out / f"{f.stem}.csv")
            if args.figure:
                check_sheet(r, out / f"{f.stem}.png", args.fs)
            status = "ok   " if len(r.leads) == 12 else "CHECK"   # a missing lead -> a person looks
            print(f"{status} {f.name}  leads={len(r.leads)}/12  flipped={r.flipped}")
        except Exception as e:  # one bad scan must not stop a batch
            print(f"FAIL  {f.name}  {e}")


if __name__ == "__main__":
    main()
