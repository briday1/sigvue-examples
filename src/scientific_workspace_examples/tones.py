"""Seekable multi-tone PSD and waterfall workspace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from workspace_browser.plugin import AnalysisWorkspace

from .sigmf import SigMFWindow, WindowedSigMF, sigmf_source
from .style import COLORS, style_figure


def analyze(data: SigMFWindow, ui) -> None:
    samples = data.samples[0]
    fft_size = min(1024, samples.size)
    hop = max(1, fft_size // 2)
    starts = np.arange(0, max(1, samples.size - fft_size + 1), hop)
    if starts.size == 0:
        starts = np.asarray([0])
    rows = np.asarray([samples[start : start + fft_size] for start in starts])
    taper = np.hanning(fft_size)
    spectra = np.fft.fftshift(np.fft.fft(rows * taper, axis=1), axes=1)
    power = np.abs(spectra / max(np.sum(taper), 1)) ** 2
    waterfall_db = 10 * np.log10(np.maximum(power, 1e-12))
    average_db = 10 * np.log10(np.maximum(np.mean(power, axis=0), 1e-12))
    frequency = np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / data.sample_rate))
    recording_time = (data.start_sample + starts + fft_size / 2) / data.sample_rate

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=(1, 2),
        vertical_spacing=0.08,
    )
    figure.add_trace(
        go.Scatter(x=frequency, y=average_db, mode="lines", line={"color": COLORS[0]}, showlegend=False),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(x=frequency, y=recording_time, z=waterfall_db, colorscale="Viridis", colorbar={"title": "dBFS"}, showscale=True),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="Average PSD (dBFS)", row=1, col=1)
    figure.update_yaxes(title_text="Recording time (s)", row=2, col=1)
    figure.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)

    with ui.tab("PSD + waterfall"):
        ui.plot(style_figure(figure, ui, "Selected multi-tone buffer"), key="tones")


def create_workspace(config=None):
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data"))
    return AnalysisWorkspace(
        identifier=str(values.get("id", "multi-tone-seek")),
        name=str(values.get("name", "Multi-Tone Seek")),
        description="Seek mode: play through a recording or choose a buffer and view its average PSD above a waterfall.",
        source=sigmf_source(root, "multiple-tones.sigmf-meta"),
        delivery=WindowedSigMF(default_buffer_seconds=0.25, playback_mode="seek"),
        analyze=analyze,
        category="spectrum monitoring",
        tags=("seek", "single-channel", "PSD", "waterfall"),
    )
