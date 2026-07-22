"""Pure Plotly figure builders for stored acoustic events."""

import plotly.graph_objects as go

from ..style import COLORS
from .models import StoredEventResults


def waveform_figure(event: StoredEventResults) -> go.Figure:
    figure = go.Figure(go.Scatter(
        x=event.waveform_time, y=event.waveform, mode="lines",
        line={"color": COLORS[0], "width": 1.5}, showlegend=False,
    ))
    figure.update_xaxes(title_text="Time within event (s)", range=[0, event.duration_seconds])
    figure.update_yaxes(title_text="Stored normalized amplitude", range=[-1.1, 1.1])
    return figure


def spectrum_figure(event: StoredEventResults) -> go.Figure:
    figure = go.Figure(go.Scatter(
        x=event.spectrum_frequency, y=event.spectrum_db, mode="lines",
        line={"color": COLORS[1], "width": 1.5}, showlegend=False,
    ))
    figure.update_xaxes(title_text="Frequency (Hz)")
    figure.update_yaxes(title_text="Stored magnitude (dB)", range=[-80, 5])
    return figure


__all__ = ["spectrum_figure", "waveform_figure"]
