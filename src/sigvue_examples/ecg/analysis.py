"""Exact ECG window calibration and annotation-derived analysis."""

from itertools import pairwise

import numpy as np

from ..plugins.wfdb import WFDBWindow
from .models import ECGProducts


def process(window: WFDBWindow, settings: None) -> ECGProducts:
    """Analyze one native window without thinning its waveform samples."""
    sample_rate = window.recording.sample_rate
    physical = window.physical_samples()
    time_seconds = (
        window.start_sample + np.arange(window.sample_count, dtype=np.float64)
    ) / sample_rate

    all_beats = tuple(
        annotation for annotation in window.recording.annotations if annotation.is_beat
    )
    rr_time: list[float] = []
    rr_seconds: list[float] = []
    rr_symbols: list[str] = []
    for previous, current in pairwise(all_beats):
        if window.start_sample <= current.sample < window.stop_sample:
            rr_time.append(current.sample / sample_rate)
            rr_seconds.append((current.sample - previous.sample) / sample_rate)
            rr_symbols.append(current.symbol)

    before = round(0.2 * sample_rate)
    after = round(0.4 * sample_rate)
    morphology_time = (
        np.arange(before + after + 1, dtype=np.float64) - before
    ) / sample_rate
    morphology: list[np.ndarray] = []
    morphology_symbols: list[str] = []
    for annotation in window.annotations:
        if not annotation.is_beat:
            continue
        center = annotation.sample - window.start_sample
        if center >= before and center + after < window.sample_count:
            morphology.append(
                physical[:, center - before : center + after + 1],
            )
            morphology_symbols.append(annotation.symbol)
    morphology_samples = (
        np.stack(morphology)
        if morphology
        else np.empty(
            (0, window.recording.channel_count, morphology_time.size),
            dtype=np.float64,
        )
    )
    return ECGProducts(
        recording=window.recording,
        start_sample=window.start_sample,
        time_seconds=time_seconds,
        physical_samples=physical,
        annotations=window.annotations,
        rr_time_seconds=np.asarray(rr_time, dtype=np.float64),
        rr_seconds=np.asarray(rr_seconds, dtype=np.float64),
        rr_symbols=tuple(rr_symbols),
        morphology_time_seconds=morphology_time,
        morphology_samples=morphology_samples,
        morphology_symbols=tuple(morphology_symbols),
        buffer_nbytes=window.buffer_nbytes,
    )


__all__ = ["process"]
