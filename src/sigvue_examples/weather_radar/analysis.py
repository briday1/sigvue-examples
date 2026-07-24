"""Exact native-gate analysis for the weather-radar example."""

from __future__ import annotations

import numpy as np

from ..plugins.nexrad import FIRST_MEASURED_CODE, NexradSequenceSelection
from .models import WeatherRadarProducts


def process(
    selection: NexradSequenceSelection,
    settings: None,
) -> WeatherRadarProducts:
    scan = selection.scan
    valid = scan.valid_gate_mask()
    measured_codes = scan.level_codes[valid & (scan.level_codes >= FIRST_MEASURED_CODE)]
    code_counts = np.bincount(measured_codes, minlength=256)
    present_codes = np.flatnonzero(code_counts[FIRST_MEASURED_CODE:]) + 2
    histogram_dbz = (
        scan.header.minimum_value_dbz
        + (present_codes - FIRST_MEASURED_CODE) * scan.header.value_increment_dbz
    )
    histogram_counts = code_counts[present_codes]
    histogram_dbz.flags.writeable = False
    histogram_counts.flags.writeable = False
    return WeatherRadarProducts(
        selection=selection,
        histogram_dbz=histogram_dbz,
        histogram_counts=histogram_counts,
        gate_counts=scan.code_counts(),
    )


__all__ = ["process"]
