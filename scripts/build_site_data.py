"""Bundle demo samples + validation metrics into docs/assets/data.js.

A plain <script> file (not JSON + fetch) so docs/index.html also works when
opened straight from disk.

    python scripts/build_site_data.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from p2p.config import LEADS   # noqa: E402

samples_dir = ROOT / "docs/assets/samples"
order = json.loads((samples_dir / "index.json").read_text())
samples = [json.loads((samples_dir / s["id"] / "result.json").read_text()) for s in order]

summary = json.loads((ROOT / "results/validation_scores.json").read_text())
scores = pd.read_csv(ROOT / "results/validation_scores.csv")
per_lead = scores.groupby("lead")[["rmse", "corr"]].mean()
# per record: average its leads first, so every record counts once per damage type
per_rec = scores.groupby(["id", "degradation"]).rmse.mean().reset_index()
metrics = {
    "records": int(scores.id.nunique()),
    "mean_rmse_mv": float(per_lead.rmse.mean()),
    "mean_corr": float(per_lead["corr"].mean()),
    "per_lead": {l: {"rmse": float(per_lead.loc[l, "rmse"]), "corr": float(per_lead.loc[l, "corr"])}
                 for l in LEADS if l in per_lead.index},
    "by_degradation": per_rec.groupby("degradation").rmse.mean().round(4).to_dict(),
    "seconds_per_page": summary["median_seconds_per_page"],
}
out = ROOT / "docs/assets/data.js"
out.write_text("window.P2P = " + json.dumps({"metrics": metrics, "samples": samples}, separators=(",", ":")) + ";\n")
print(json.dumps({k: v for k, v in metrics.items() if k != "per_lead"}, indent=1))
print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
