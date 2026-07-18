"""Windowed constellation and eye-diagram workspace for digital communications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from workspace_browser.plugin import AnalysisContext, AnalysisWorkspace, DataDelivery, DataResource, DirectorySource

from .sigmf import SigMFRecording, load_metadata, load_recording
from .style import COLORS, style_figure


@dataclass(frozen=True)
class CommsWindow:
    """The exact recording interval selected by this workspace's delivery policy."""

    recording: SigMFRecording
    start_sample: int
    samples: np.ndarray

    @property
    def sample_rate(self) -> float:
        return self.recording.sample_rate

    @property
    def time_seconds(self) -> np.ndarray:
        return (self.start_sample + np.arange(self.samples.shape[1])) / self.sample_rate


class WindowedCommsDelivery(DataDelivery[SigMFRecording, CommsWindow]):
    """Select a short interval over a decimated received-power overview."""

    def prepare(self, recording: SigMFRecording, ui: AnalysisContext) -> CommsWindow:
        overview = ui.once(
            f"comms-power-overview:{recording.metadata_path}",
            lambda: _power_overview(recording),
        )
        start_seconds, end_seconds = ui.windowed(
            duration=recording.duration_seconds,
            default_window=min(0.03, recording.duration_seconds),
            minimum_window=min(0.005, recording.duration_seconds),
            step=min(0.005, recording.duration_seconds),
            overview=overview,
            overview_label="Received power",
        )
        start = round(start_seconds * recording.sample_rate)
        count = max(1, round((end_seconds - start_seconds) * recording.sample_rate))
        return CommsWindow(recording, start, recording.read(start, count))


def _power_overview(recording: SigMFRecording) -> np.ndarray:
    samples = recording.read(0, recording.sample_count)[0]
    block_count = min(200, samples.size)
    block_size = max(1, samples.size // block_count)
    blocks = samples[: block_count * block_size].reshape(block_count, block_size)
    return 10 * np.log10(np.maximum(np.mean(np.abs(blocks) ** 2, axis=1), 1e-12))


def analyze(data: CommsWindow, ui: AnalysisContext) -> None:
    samples = data.samples[0]
    metadata = data.recording.metadata["global"]
    modulation = str(metadata.get("examples:modulation", "Digital modulation"))
    symbol_rate = float(metadata["examples:symbol_rate"])
    carrier_hz = float(metadata["examples:carrier_hz"])
    baseband = samples * np.exp(-1j * 2 * np.pi * carrier_hz * data.time_seconds)
    samples_per_symbol = max(1, round(data.sample_rate / symbol_rate))
    alignment = (-data.start_sample) % samples_per_symbol
    aligned = baseband[alignment:]

    symbol_count = aligned.size // samples_per_symbol
    symbols = aligned[: symbol_count * samples_per_symbol].reshape(symbol_count, samples_per_symbol).mean(axis=1)
    constellation = go.Figure(go.Scattergl(
        x=symbols.real,
        y=symbols.imag,
        mode="markers",
        marker={"color": COLORS[0], "size": 6, "opacity": 0.55},
        showlegend=False,
    ))
    constellation_limit = float(metadata.get("examples:constellation_limit", 0.8))
    constellation.update_xaxes(
        title_text="In-phase",
        range=[-constellation_limit, constellation_limit],
        autorange=False,
        scaleanchor="y",
        scaleratio=1,
    )
    constellation.update_yaxes(
        title_text="Quadrature",
        range=[-constellation_limit, constellation_limit],
        autorange=False,
    )

    eye_length = 2 * samples_per_symbol
    eye_count = min(160, max(0, aligned.size // samples_per_symbol - 1))
    eye_segments = np.asarray([
        aligned[index * samples_per_symbol : index * samples_per_symbol + eye_length]
        for index in range(eye_count)
    ])
    eye_time = np.arange(eye_length) / samples_per_symbol
    eye_x = np.concatenate([np.append(eye_time, np.nan) for _ in range(eye_count)])
    eye = go.Figure()
    if eye_count:
        eye.add_trace(go.Scattergl(
            x=eye_x,
            y=np.concatenate([np.append(segment.real, np.nan) for segment in eye_segments]),
            name="I",
            mode="lines",
            line={"color": COLORS[0], "width": 1},
        ))
        eye.add_trace(go.Scattergl(
            x=eye_x,
            y=np.concatenate([np.append(segment.imag, np.nan) for segment in eye_segments]),
            name="Q",
            mode="lines",
            line={"color": COLORS[1], "width": 1},
        ))
    eye_limit = float(metadata.get("examples:eye_limit", constellation_limit))
    eye.update_xaxes(title_text="Symbol periods", range=[0, 2], autorange=False)
    eye.update_yaxes(title_text="Amplitude", range=[-eye_limit, eye_limit], autorange=False)

    ui.stat("Modulation", modulation)
    ui.stat("Symbol rate", f"{symbol_rate / 1e3:g} ksym/s")
    with ui.tab("Constellation"):
        ui.plot(style_figure(constellation, ui.theme, f"{modulation} constellation"), key="constellation")
    with ui.tab("Eye diagram"):
        ui.plot(style_figure(eye, ui.theme, f"{modulation} eye diagram"), key="eye")


def _describe_recording(metadata_path: Path) -> DataResource:
    metadata = load_metadata(metadata_path)
    global_metadata = metadata["global"]
    modulation = str(global_metadata.get("examples:modulation") or global_metadata.get("core:description") or metadata_path.stem)
    return DataResource(
        identifier=metadata_path.name.removesuffix(".sigmf-meta"),
        title=modulation,
        source=metadata_path,
        subtitle=f"{float(global_metadata['core:sample_rate']):g} samples/s",
        timestamp=datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=timezone.utc),
        tags=("sigmf", str(global_metadata["core:datatype"]), modulation.lower()),
    )


def create_workspace(config=None):
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data/comms"))
    return AnalysisWorkspace(
        identifier=str(values.get("id", "digital-comms")),
        name=str(values.get("name", "Digital Communications")),
        description="Windowed mode: compare file-backed QPSK and 16-QAM recordings with constellation and eye-diagram views.",
        source=DirectorySource(
            root,
            pattern="*.sigmf-meta",
            loader=load_recording,
            describe=_describe_recording,
        ),
        delivery=WindowedCommsDelivery(),
        analyze=analyze,
        category="digital communications",
        tags=("windowed", "qpsk", "16-qam", "constellation", "eye diagram"),
    )
