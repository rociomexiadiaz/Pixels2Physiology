"""How close is the recovered signal to the original recording?

Before scoring, the prediction is shifted by up to +/-0.2 s and given a constant
vertical offset, so a trace that is right but starts a few pixels late, or sits
on a different baseline, is not penalised for that alone. RMSE is in mV.
"""
import numpy as np
from scipy.signal import correlate


def aligned_scores(pred, target, fs, max_shift_sec=0.2, return_aligned=False):
    """Returns dict(rmse, lag_s, offset_mv, corr), plus the aligned arrays if asked."""
    pred = np.asarray(pred, float)
    target = np.asarray(target, float)
    target = target[~np.isnan(target)]
    L = min(len(pred), len(target))
    pred, target = pred[:L], target[:L]

    max_shift = int(max_shift_sec * fs)
    corr = correlate(target, pred, mode="full")
    lags = np.arange(-len(pred) + 1, len(target))
    valid = np.where((lags >= -max_shift) & (lags <= max_shift))[0]
    best_lag = int(lags[valid][np.argmax(corr[valid])])

    # correlate(target, pred)[lag] peaks where target[n + lag] ~ pred[n]
    lag = best_lag
    if lag > 0:
        t = target[lag:]
        p = pred[:len(t)]
    elif lag < 0:
        p = pred[-lag:]
        t = target[:len(p)]
    else:
        p, t = pred, target

    offset = float(np.mean(t - p))
    rmse = float(np.sqrt(np.mean((p + offset - t) ** 2)))

    pn = (p - p.mean()) / (p.std() + 1e-8)
    tn = (t - t.mean()) / (t.std() + 1e-8)
    out = {"rmse": rmse, "lag_s": best_lag / fs, "offset_mv": offset, "corr": float(np.mean(pn * tn))}
    if return_aligned:
        out["pred_aligned"], out["true_aligned"] = p + offset, t
    return out
