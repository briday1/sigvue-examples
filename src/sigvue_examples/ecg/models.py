"""Typed products for the annotated ECG workspace."""

from dataclasses import dataclass

import numpy as np

from ..plugins.wfdb import WFDBAnnotation, WFDBRecording


@dataclass(frozen=True)
class ECGProducts:
    """Exact calibrated samples and derived annotation products for one window."""

    recording: WFDBRecording
    start_sample: int
    time_seconds: np.ndarray
    physical_samples: np.ndarray
    annotations: tuple[WFDBAnnotation, ...]
    rr_time_seconds: np.ndarray
    rr_seconds: np.ndarray
    rr_symbols: tuple[str, ...]
    morphology_time_seconds: np.ndarray
    morphology_samples: np.ndarray
    morphology_symbols: tuple[str, ...]
    buffer_nbytes: int

    @property
    def stop_sample(self) -> int:
        return self.start_sample + self.physical_samples.shape[1]

    @property
    def duration_seconds(self) -> float:
        return self.physical_samples.shape[1] / self.recording.sample_rate

    @property
    def beat_count(self) -> int:
        return sum(annotation.is_beat for annotation in self.annotations)

    @property
    def annotation_rows(self) -> list[dict[str, object]]:
        return [
            {
                "Time (s)": annotation.time_seconds(
                    self.recording.sample_rate,
                ),
                "Sample": annotation.sample,
                "Symbol": annotation.symbol,
                "Type": annotation.description,
                "Channel": annotation.channel,
                "Note": annotation.auxiliary_note,
            }
            for annotation in self.annotations
        ]

    @property
    def metadata_rows(self) -> list[dict[str, object]]:
        header = self.recording.header
        rows: list[dict[str, object]] = [
            {"Field": "Record", "Value": header.record_name},
            {"Field": "Sampling rate", "Value": f"{header.sample_rate:g} Hz"},
            {"Field": "Samples", "Value": f"{header.sample_count:,}"},
            {
                "Field": "Duration",
                "Value": f"{header.duration_seconds / 60:g} min",
            },
            {
                "Field": "Reference annotations",
                "Value": f"{len(self.recording.annotations):,}",
            },
        ]
        for index, channel in enumerate(header.channels, start=1):
            rows.extend(
                (
                    {
                        "Field": f"Lead {index}",
                        "Value": channel.name,
                    },
                    {
                        "Field": f"Lead {index} calibration",
                        "Value": (
                            f"{channel.gain:g} ADC/{channel.units}; "
                            f"baseline {channel.baseline}"
                        ),
                    },
                    {
                        "Field": f"Lead {index} native checksum",
                        "Value": channel.checksum,
                    },
                )
            )
        if header.comments:
            rows.append(
                {
                    "Field": "Header notes",
                    "Value": " · ".join(header.comments),
                }
            )
        return rows
