"""Minimal SigMF discovery plus whole-file and windowed delivery policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from workspace_browser.plugin import AnalysisContext, DataResource, DirectorySource


@dataclass(frozen=True)
class SigMFRecording:
    metadata_path: Path
    data_path: Path
    sample_rate: float
    channel_count: int
    sample_count: int
    metadata: dict[str, object]

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate

    def read(self, start: int, count: int) -> np.ndarray:
        """Read only the requested interleaved cf32_le frames."""
        start = min(self.sample_count, max(0, start))
        count = min(max(0, count), self.sample_count - start)
        scalars_per_frame = self.channel_count * 2
        with self.data_path.open("rb") as stream:
            stream.seek(start * scalars_per_frame * 4)
            scalars = np.fromfile(stream, dtype="<f4", count=count * scalars_per_frame)
        frames = scalars.reshape(-1, self.channel_count, 2)
        return np.asarray(frames[..., 0] + 1j * frames[..., 1], dtype=np.complex64).T


@dataclass(frozen=True)
class SigMFWindow:
    recording: SigMFRecording
    start_sample: int
    samples: np.ndarray

    @property
    def sample_rate(self) -> float:
        return self.recording.sample_rate

    @property
    def time_seconds(self) -> np.ndarray:
        return (self.start_sample + np.arange(self.samples.shape[1])) / self.sample_rate


class WindowedSigMF:
    """Framework delivery policy that turns playback time into one file read."""

    def __init__(self, *, default_buffer_seconds: float, playback_mode: str = "seek") -> None:
        self.default_buffer_seconds = default_buffer_seconds
        self.playback_mode = playback_mode

    def prepare(self, recording: SigMFRecording, ui: AnalysisContext) -> SigMFWindow:
        buffer_seconds = ui.number(
            "buffer_seconds",
            label="Buffer (s)",
            default=self.default_buffer_seconds,
            minimum=1 / recording.sample_rate,
            maximum=recording.duration_seconds,
            step=self.default_buffer_seconds / 4,
        )
        count = min(recording.sample_count, max(1, round(buffer_seconds * recording.sample_rate)))
        duration = max(0.0, (recording.sample_count - count) / recording.sample_rate)
        position = ui.playback(
            mode=self.playback_mode,
            duration=duration,
            step=max(1 / recording.sample_rate, buffer_seconds / 4),
            refresh_interval=0.2,
            loop=True,
        )
        start = min(round(position * recording.sample_rate), recording.sample_count - count)
        return SigMFWindow(recording, start, recording.read(start, count))


class WholeSigMF:
    """Framework delivery policy that gives analysis the complete recording."""

    def prepare(self, recording: SigMFRecording, ui: AnalysisContext) -> SigMFWindow:
        ui.playback(mode="static")
        return SigMFWindow(recording, 0, recording.read(0, recording.sample_count))


def sigmf_source(directory: Path, filename: str) -> DirectorySource:
    return DirectorySource(directory, pattern=filename, loader=load_recording, describe=describe_recording)


def describe_recording(metadata_path: Path) -> DataResource:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    global_metadata = metadata["global"]
    channels = int(global_metadata.get("core:num_channels", 1))
    return DataResource(
        identifier=metadata_path.name.removesuffix(".sigmf-meta"),
        title=str(global_metadata.get("core:description", metadata_path.stem)),
        source=metadata_path,
        subtitle=f"{channels} channel{'s' if channels != 1 else ''} · {float(global_metadata['core:sample_rate']):g} samples/s",
        timestamp=datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=timezone.utc),
        tags=("sigmf", "cf32_le"),
    )


def load_recording(metadata_path: Path) -> SigMFRecording:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    global_metadata = metadata["global"]
    if global_metadata.get("core:datatype") != "cf32_le":
        raise ValueError("Minimal examples require cf32_le SigMF data")
    channel_count = int(global_metadata.get("core:num_channels", 1))
    data_path = metadata_path.with_name(metadata_path.name.removesuffix(".sigmf-meta") + ".sigmf-data")
    sample_count = data_path.stat().st_size // (channel_count * 2 * 4)
    return SigMFRecording(
        metadata_path,
        data_path,
        float(global_metadata["core:sample_rate"]),
        channel_count,
        sample_count,
        metadata,
    )


def spectrum(samples: np.ndarray, sample_rate: float, *, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    size = samples.size
    taper = np.hanning(size) if window == "hann" else np.ones(size)
    values = np.fft.fftshift(np.fft.fft(samples * taper))
    frequency = np.fft.fftshift(np.fft.fftfreq(size, 1 / sample_rate))
    magnitude = 20 * np.log10(np.maximum(np.abs(values) / max(np.sum(taper), 1), 1e-9))
    return frequency, magnitude
