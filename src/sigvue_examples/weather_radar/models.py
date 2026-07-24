"""Models exchanged by the weather-radar analysis and presentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..plugins.nexrad import NexradLevel3Radial, NexradSequenceSelection


@dataclass(frozen=True)
class WeatherRadarProducts:
    selection: NexradSequenceSelection
    histogram_dbz: np.ndarray
    histogram_counts: np.ndarray
    gate_counts: dict[str, int]

    @property
    def scan(self) -> NexradLevel3Radial:
        return self.selection.scan


__all__ = ["WeatherRadarProducts"]
