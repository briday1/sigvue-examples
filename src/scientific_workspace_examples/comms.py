"""Windowed QPSK constellation and eye-diagram workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from workspace_browser.plugin import AnalysisWorkspace

from .sigmf import SigMFRecording, SigMFWindow, sigmf_source
from .style import COLORS, style_figure


class WindowedQpskDelivery:
    """Select a small interval over a full-record received-power summary."""

    def prepare(self, recording: SigMFRecording, ui) -> SigMFWindow:
        overview = ui.once("qpsk-power-overview", lambda: _power_overview(recording))
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
        return SigMFWindow(recording, start, recording.read(start, count))


def _power_overview(recording: SigMFRecording) -> np.ndarray:
    samples = recording.read(0, recording.sample_count)[0]
    block_count = min(200, samples.size)
    block_size = max(1, samples.size // block_count)
    blocks = samples[: block_count * block_size].reshape(block_count, block_size)
    return 10 * np.log10(np.maximum(np.mean(np.abs(blocks) ** 2, axis=1), 1e-12))


def analyze(data: SigMFWindow, ui) -> None:
    samples = data.samples[0]
    metadata = data.recording.metadata["global"]
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
    # The generated recording's symbol centers stay within about +/-0.66.
    # Fixed symmetric limits leave a small margin without making the clusters
    # look artificially compressed as the selected window moves.
    constellation.update_xaxes(
        title_text="In-phase",
        range=[-0.75, 0.75],
        autorange=False,
        scaleanchor="y",
        scaleratio=1,
    )
    constellation.update_yaxes(title_text="Quadrature", range=[-0.75, 0.75], autorange=False)

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
    # Keep the eye geometry stable while the framework slides the selected
    # window through the recording. These limits are a presentation choice for
    # this normalized QPSK example, not part of windowed delivery.
    eye.update_xaxes(title_text="Symbol periods", range=[0, 2], autorange=False)
    eye.update_yaxes(title_text="Amplitude", range=[-1, 1], autorange=False)

    with ui.tab("Constellation"):
        ui.plot(style_figure(constellation, ui, "QPSK constellation"), key="constellation")
    with ui.tab("Eye diagram"):
        ui.plot(style_figure(eye, ui, "QPSK eye diagram"), key="eye")


def create_workspace(config=None):
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data"))
    return AnalysisWorkspace(
        identifier=str(values.get("id", "qpsk-windowed")),
        name=str(values.get("name", "QPSK Windowed Analysis")),
        description="Windowed mode: drag or resize a short interval and inspect its QPSK constellation and eye diagram.",
        source=sigmf_source(root, "qpsk.sigmf-meta"),
        delivery=WindowedQpskDelivery(),
        analyze=analyze,
        category="digital communications",
        tags=("windowed", "single-channel", "constellation", "eye diagram"),
    )
