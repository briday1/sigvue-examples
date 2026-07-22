"""Recover symbol samples and eye traces from the selected recording window."""

import numpy as np

from sigvue.plugin import Analysis

from .models import CommsProducts, CommsWindow


def process(window: CommsWindow, settings: None) -> CommsProducts:
    metadata = window.recording.metadata["global"]
    modulation = str(metadata.get("example:modulation", metadata.get("examples:modulation", "Unknown")))
    symbol_rate = float(metadata.get("examples:symbol_rate", 0.0) or 0.0)
    samples_per_symbol = int(metadata.get(
        "example:samples_per_symbol",
        round(window.recording.sample_rate / symbol_rate) if symbol_rate > 0 else 8,
    ))
    sample_offset = int(metadata.get("example:sample_offset", samples_per_symbol // 2))
    carrier_hz = float(metadata.get("examples:carrier_hz", 0.0) or 0.0)
    absolute_samples = window.start_sample + np.arange(window.samples.size)
    samples = window.samples * np.exp(
        -2j * np.pi * carrier_hz * absolute_samples / window.recording.sample_rate
    )

    alignment = (sample_offset - window.start_sample) % samples_per_symbol
    symbols = samples[alignment::samples_per_symbol]

    eye_length = samples_per_symbol * 2
    eye_starts = np.arange(alignment, max(alignment, samples.size - eye_length), samples_per_symbol)
    eye_starts = eye_starts[:180]
    eye_segments = np.asarray([samples[start : start + eye_length] for start in eye_starts])
    eye_time = np.arange(eye_length) / samples_per_symbol

    constellation_extent = float(metadata.get(
        "example:constellation_limit",
        metadata.get("examples:constellation_limit", 0.9),
    ))
    eye_extent = float(metadata.get("example:eye_limit", metadata.get("examples:eye_limit", 0.95)))
    return CommsProducts(
        modulation,
        samples_per_symbol,
        symbols,
        eye_time,
        eye_segments,
        constellation_extent,
        eye_extent,
        window.start_seconds,
        window.duration_seconds,
        window.samples.nbytes,
    )


class CommsAnalysis(Analysis[CommsWindow, None, CommsProducts]):
    """Framework analysis object for a selected communications window."""

    def process(self, window: CommsWindow, settings: None) -> CommsProducts:
        return process(window, settings)
