"""Pixels2Physiology - turn a photo or scan of a paper ECG into 12-lead signals.

    from p2p import Pipeline
    result = Pipeline.from_pretrained().run("scan.png")
    result.signals_dataframe(fs=500, n_samples=5000).to_csv("ecg.csv")
"""
from .pipeline import Pipeline, PipelineResult

__all__ = ["Pipeline", "PipelineResult"]
