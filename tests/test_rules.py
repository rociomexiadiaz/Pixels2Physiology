"""Fast checks on the rule-based steps (no model weights needed).

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from p2p import orient                                  # noqa: E402
from p2p.config import CANVAS_H, CANVAS_W, MV_PER_PX    # noqa: E402
from p2p.digitise import extract_ecg_trace, resample_to_length, strip_to_leads   # noqa: E402
from p2p.metrics import aligned_scores                  # noqa: E402
from p2p.preprocess import order_corners_clockwise, resize_with_padding          # noqa: E402


def test_padding_keeps_aspect_and_canvas():
    out = resize_with_padding(np.full((1000, 1000), 255, np.uint8))
    assert out.shape == (CANVAS_H, CANVAS_W)
    assert (out[:, :100] == 0).all() and (out[:, CANVAS_W // 2] == 255).all()


def test_corner_order():
    pts = [[10, 90], [90, 10], [10, 10], [90, 90]]
    assert order_corners_clockwise(pts).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_trace_extraction_reads_peak_height():
    mask = np.zeros((384, 200), np.uint8)
    mask[200, :] = 1                 # baseline
    mask[122:200, 100] = 1           # a 78 px spike = 1 mV
    _, volts, base = extract_ecg_trace(mask)
    assert base == 200
    assert abs(volts[100] - 78 * MV_PER_PX) < 1e-9 and volts[0] == 0


def test_short_strip_splits_into_four_leads():
    mask = np.zeros((384, 400), np.uint8); mask[190, :] = 1
    leads = strip_to_leads(mask, 0, 5000)
    assert list(leads) == ["I", "aVR", "V1", "V4"] and all(len(v) == 1250 for v in leads.values())


def test_metric_ignores_small_time_shift():
    t = np.sin(np.linspace(0, 20, 1000))
    assert aligned_scores(np.roll(t, 5), t, fs=100)["rmse"] < 0.05


def test_metric_ignores_constant_offset():
    t = np.sin(np.linspace(0, 20, 1000))
    assert aligned_scores(t + 0.3, t, fs=100)["rmse"] < 1e-9


def test_orientation_prefers_page_with_real_traces():
    good = np.zeros((384, 100)); good[190, :] = 1; good[100:190, 50] = 1
    flat = np.zeros((384, 100)); flat[190, :] = 1
    assert orient.orientation_score([good] * 4) > orient.orientation_score([good, flat, flat, good])
    assert orient.needs_flip([good, flat, good, good])


def test_resample_length():
    assert len(resample_to_length(np.arange(10.0), 37)) == 37
