"""Domain models for stored acoustic-event collections."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredEventResults:
    identifier: str
    label: str
    start_seconds: float
    duration_seconds: float
    confidence: float
    waveform_time: tuple[float, ...]
    waveform: tuple[float, ...]
    spectrum_frequency: tuple[float, ...]
    spectrum_db: tuple[float, ...]

    @property
    def buffer_nbytes(self) -> int:
        """Logical numeric payload delivered to the event analysis."""
        return 8 * (
            len(self.waveform_time) + len(self.waveform)
            + len(self.spectrum_frequency) + len(self.spectrum_db)
        )


@dataclass(frozen=True)
class AcousticEventCollection:
    path: Path
    duration_seconds: float
    events: tuple[StoredEventResults, ...]
