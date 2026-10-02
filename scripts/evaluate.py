"""Score the pipeline on the held-out validation set.

Reproduces the notebook split (seeded), runs every validation record at its
assigned degradation, and reports aligned RMSE (mV) and correlation per lead.

One pass scores four variants, so the paper's number and the corrected one
come from the same model outputs:
  orientation  notebook | compare   (see p2p/orient.py)
  metric       paper    | fixed     (see p2p/metrics.py)

    python scripts/evaluate.py --data /path/to/physionet-ecg-image-digitization
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from p2p import Pipeline, orient                          # noqa: E402
from p2p.config import DEGRADATIONS, STRIP_LEADS          # noqa: E402
from p2p.digitise import clean_mask, strip_to_leads       # noqa: E402
from p2p.metrics import aligned_scores                    # noqa: E402
from p2p.preprocess import load_gray, resize_with_padding # noqa: E402

LEAD_ORDER = [l for names in STRIP_LEADS.values() for l in names][:12]


def validation_split(meta):
    train_grid, _ = train_test_split(meta, test_size=0.1, random_state=42)
    _, val_grid = train_test_split(train_grid, test_size=0.2, random_state=42)
    rng = np.random.default_rng(seed=0)
    pool = [3, 4, 5, 6, 9, 10, 11, 12]
    val_grid = val_grid.copy()
    val_grid["suffix"] = [int(rng.choice(pool)) for _ in range(len(val_grid))]
    return val_grid


def score_masks(strip_masks, truth, fs, n, paper):
    """Per-lead scores. Lead II is scored on both its 2.5 s and 10 s copies, as in the paper."""
    q = n // 4
    rows = []
    for i, m in enumerate(strip_masks):
        for j, (lead, v) in enumerate(strip_to_leads(clean_mask(m), i, n).items()):
            gt = truth[lead].values
            if i != 3:
                gt = gt[j * q:(j + 1) * q]
            rows.append({"lead": lead, "strip": i + 1, **aligned_scores(v, gt, fs, paper=paper)})
    return rows


def summarise(df):
    per_lead = df.groupby("lead")[["rmse", "corr"]].mean()
    per_rec = df.groupby(["id", "degradation"]).rmse.mean().reset_index()
    return {"records": int(df.id.nunique()),
            "mean_rmse_mv": round(float(per_lead.rmse.mean()), 4),
            "mean_corr": round(float(per_lead["corr"].mean()), 4),
            "flipped_pages": int(df.groupby("id").flipped.first().sum()),
            "per_lead": per_lead.loc[[l for l in LEAD_ORDER if l in per_lead.index]].round(4).to_dict("index"),
            "by_degradation_rmse": per_rec.groupby("degradation").rmse.mean().round(4).to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/validation_scores.csv")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    data = Path(args.data)
    val = validation_split(pd.read_csv(data / "train.csv"))
    if args.limit:
        val = val.head(args.limit)

    pipe = Pipeline.from_pretrained()
    rows, seconds = [], []
    for k, r in enumerate(val.itertuples(), 1):
        rid = str(r.id)
        t0 = time.perf_counter()
        page = resize_with_padding(load_gray(data / "train" / rid / f"{rid}-{r.suffix:04d}.png"))
        first = pipe._once(page)[4]
        fires = orient.needs_flip(first)
        second = pipe._once(orient.rotate_180(page))[4] if fires else None
        better = fires and orient.orientation_score(second) > orient.orientation_score(first)
        variants = {"notebook": (second, True) if fires else (first, False),
                    "compare": (second, True) if better else (first, False)}
        seconds.append(time.perf_counter() - t0)

        truth = pd.read_csv(data / "train" / rid / f"{rid}.csv")
        for orient_mode, (masks, flipped) in variants.items():
            for metric in ("paper", "fixed"):
                for row in score_masks(masks, truth, r.fs, r.sig_len, paper=metric == "paper"):
                    rows.append({"id": rid, "suffix": r.suffix, "degradation": DEGRADATIONS[r.suffix],
                                 "orientation": orient_mode, "metric": metric, "flipped": flipped,
                                 "rule_fired": fires, **row})
        print(f"[{k}/{len(val)}] {rid} {DEGRADATIONS[r.suffix]} fired={fires} kept_flip={better}", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = {f"{o}_orientation+{m}_metric": summarise(g) for (o, m), g in df.groupby(["orientation", "metric"])}
    summary["median_seconds_per_page_incl_second_pass"] = round(float(np.median(seconds)), 2)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "per_lead"})
                      for k, v in summary.items()}, indent=1))
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
