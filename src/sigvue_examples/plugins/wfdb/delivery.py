"""Reusable exact window delivery and ECG overview generation for WFDB."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from sigvue.plugin import Delivery, DeliveryContext, TimeUnit

from .annotations import WFDBAnnotation
from .recording import WFDBRecording


@dataclass(frozen=True)
class WFDBWindow:
    """A native integer WFDB window delivered channel-first."""

    recording: WFDBRecording
    start_sample: int
    digital_samples: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.start_sample, bool) or not isinstance(
            self.start_sample, int
        ):
            raise TypeError("start_sample must be an integer")
        if not 0 <= self.start_sample <= self.recording.sample_count:
            raise ValueError("Window start is outside the recording")
        if (
            not isinstance(self.digital_samples, np.ndarray)
            or self.digital_samples.ndim != 2
            or self.digital_samples.dtype.kind != "i"
        ):
            raise ValueError(
                "WFDB window samples must be a channel-first integer array"
            )
        if self.digital_samples.shape[0] != self.recording.channel_count:
            raise ValueError("WFDB window channel count does not match record")
        if self.stop_sample > self.recording.sample_count:
            raise ValueError("WFDB window extends beyond the recording")

    @property
    def sample_count(self) -> int:
        return int(self.digital_samples.shape[1])

    @property
    def stop_sample(self) -> int:
        return self.start_sample + self.sample_count

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.recording.sample_rate

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.recording.sample_rate

    @property
    def buffer_nbytes(self) -> int:
        return int(self.digital_samples.nbytes)

    @property
    def annotations(self) -> tuple[WFDBAnnotation, ...]:
        return self.recording.annotations_between(
            self.start_sample,
            self.stop_sample,
        )

    def physical_samples(self) -> np.ndarray:
        """Apply header calibration while preserving the native integer buffer."""
        baselines = np.asarray(
            [channel.baseline for channel in self.recording.header.channels],
            dtype=np.float64,
        )[:, None]
        gains = np.asarray(
            [channel.gain for channel in self.recording.header.channels],
            dtype=np.float64,
        )[:, None]
        return (self.digital_samples.astype(np.float64) - baselines) / gains


def peak_to_peak_overview(
    recording: WFDBRecording,
    *,
    bins: int = 300,
    channel: int = 0,
) -> np.ndarray:
    """Compute exact per-bin peak-to-peak amplitude with bounded reads."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be positive")
    if (
        isinstance(channel, bool)
        or not isinstance(channel, int)
        or not 0 <= channel < recording.channel_count
    ):
        raise ValueError("overview channel is outside the recording")
    count = min(bins, recording.sample_count)
    edges = np.linspace(
        0,
        recording.sample_count,
        count + 1,
        dtype=np.int64,
    )
    values = np.empty(count, dtype=np.float64)
    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        samples = recording.read_physical(
            int(start),
            int(stop - start),
        )[channel]
        values[index] = (
            float(np.max(samples) - np.min(samples)) if samples.size else 0.0
        )
    return values


class WindowedWFDBDelivery(Delivery[WFDBRecording, WFDBWindow]):
    """Configurable window selection for any supported local WFDB record."""

    def __init__(
        self,
        *,
        default_window: float = 10.0,
        minimum_window: float = 2.0,
        step: float = 1.0,
        overview_bins: int = 300,
        overview_channel: int = 0,
        overview_label: str | None = None,
        time_unit: TimeUnit = "s",
        cache_key: str = "wfdb-peak-to-peak",
    ) -> None:
        if (
            not all(isfinite(value) for value in (default_window, minimum_window, step))
            or min(default_window, minimum_window, step) <= 0
        ):
            raise ValueError(
                "WFDB window durations and step must be finite and positive"
            )
        if (
            isinstance(overview_bins, bool)
            or not isinstance(overview_bins, int)
            or overview_bins < 1
        ):
            raise ValueError("overview_bins must be positive")
        if (
            isinstance(overview_channel, bool)
            or not isinstance(overview_channel, int)
            or overview_channel < 0
        ):
            raise ValueError("overview_channel must be non-negative")
        if not cache_key:
            raise ValueError("cache_key cannot be empty")
        self.default_window = default_window
        self.minimum_window = minimum_window
        self.step = step
        self.overview_bins = overview_bins
        self.overview_channel = overview_channel
        self.overview_label = overview_label
        self.time_unit = time_unit
        self.cache_key = cache_key

    def prepare(
        self,
        recording: WFDBRecording,
        ui: DeliveryContext,
    ) -> WFDBWindow:
        if self.overview_channel >= recording.channel_count:
            raise ValueError("overview channel is outside the recording")
        overview = ui.once(
            f"{self.cache_key}:{recording.header.path}",
            lambda: peak_to_peak_overview(
                recording,
                bins=self.overview_bins,
                channel=self.overview_channel,
            ),
        )
        channel_name = recording.channel_names[self.overview_channel]
        channel_units = recording.header.channels[self.overview_channel].units
        start_seconds, stop_seconds = ui.windowed(
            duration=recording.duration_seconds,
            default_window=min(
                self.default_window,
                recording.duration_seconds,
            ),
            minimum_window=min(
                self.minimum_window,
                recording.duration_seconds,
            ),
            step=min(self.step, recording.duration_seconds),
            overview=overview,
            overview_label=(
                self.overview_label or f"Peak-to-peak {channel_name} ({channel_units})"
            ),
            time_unit=self.time_unit,
        )
        start_sample = min(
            recording.sample_count - 1,
            round(start_seconds * recording.sample_rate),
        )
        stop_sample = min(
            recording.sample_count,
            max(
                start_sample + 1,
                round(stop_seconds * recording.sample_rate),
            ),
        )
        return WFDBWindow(
            recording=recording,
            start_sample=start_sample,
            digital_samples=recording.read_digital(
                start_sample,
                stop_sample - start_sample,
            ),
        )


__all__ = [
    "WFDBWindow",
    "WindowedWFDBDelivery",
    "peak_to_peak_overview",
]
