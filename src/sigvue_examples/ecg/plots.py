"""Pure Plotly builders for ECG, RR, and beat-morphology views."""

from html import escape

import numpy as np
import plotly.graph_objects as go

from sigvue.plugin import TraceStyle

from ..style import ORANGE, TEAL
from .models import ECGProducts


def waveform_figure(
    products: ECGProducts,
    channel: int,
    waveform_style: TraceStyle,
    annotation_style: TraceStyle,
) -> go.Figure:
    """Plot every delivered sample and exact reference annotation location."""
    channel_info = products.recording.header.channels[channel]
    values = products.physical_samples[channel]
    figure = go.Figure(
        go.Scattergl(
            x=products.time_seconds,
            y=values,
            mode=waveform_style.mode,
            line=waveform_style.line,
            marker=waveform_style.plotly_marker,
            name=channel_info.name,
            showlegend=False,
        )
    )
    offsets = np.asarray(
        [
            annotation.sample - products.start_sample
            for annotation in products.annotations
        ],
        dtype=np.int64,
    )
    if offsets.size:
        hover = []
        symbols = []
        for annotation in products.annotations:
            details = [
                f"<b>{escape(annotation.description)}</b>",
                f"Symbol: {escape(annotation.symbol)}",
                f"Sample: {annotation.sample:,}",
                (
                    "Time: "
                    f"{annotation.time_seconds(products.recording.sample_rate):.6f} s"
                ),
            ]
            if annotation.auxiliary_note:
                details.append(escape(annotation.auxiliary_note))
            hover.append("<br>".join(details))
            symbols.append(annotation.symbol)
        figure.add_trace(
            go.Scatter(
                x=products.time_seconds[offsets],
                y=values[offsets],
                mode="markers+text",
                marker={
                    **annotation_style.plotly_marker,
                    "size": max(8.0, annotation_style.width * 4),
                },
                text=symbols,
                textposition="top center",
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
                name="Reference annotations",
                showlegend=False,
                opacity=annotation_style.opacity,
            )
        )
    figure.update_xaxes(
        title_text="Elapsed time (s)",
        range=[
            products.start_sample / products.recording.sample_rate,
            products.stop_sample / products.recording.sample_rate,
        ],
        autorange=False,
    )
    figure.update_yaxes(
        title_text=f"{channel_info.name} ({channel_info.units})",
    )
    return figure


def rr_figure(products: ECGProducts) -> go.Figure:
    """Show exact beat-to-beat intervals and their annotation symbols."""
    hover = [
        f"Beat: {escape(symbol)}<br>RR: {interval:.6f} s"
        for symbol, interval in zip(
            products.rr_symbols,
            products.rr_seconds,
            strict=True,
        )
    ]
    figure = go.Figure(
        go.Scatter(
            x=products.rr_time_seconds,
            y=products.rr_seconds,
            mode="lines+markers",
            line={"color": TEAL, "width": 1.4},
            marker={"color": ORANGE, "size": 6},
            customdata=hover,
            hovertemplate="%{customdata}<br>Time: %{x:.6f} s<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_xaxes(title_text="Elapsed time (s)")
    figure.update_yaxes(title_text="RR interval (s)")
    return figure


def morphology_figure(
    products: ECGProducts,
    channel: int,
) -> go.Figure:
    """Overlay every complete beat cutout present in the delivered window."""
    figure = go.Figure()
    for index, (segment, symbol) in enumerate(
        zip(
            products.morphology_samples[:, channel],
            products.morphology_symbols,
            strict=True,
        )
    ):
        figure.add_trace(
            go.Scatter(
                x=products.morphology_time_seconds,
                y=segment,
                mode="lines",
                line={"color": TEAL, "width": 0.8},
                opacity=0.22,
                zorder=0,
                name=f"Beat {index + 1} ({symbol})",
                showlegend=False,
                hovertemplate=(
                    f"Beat {index + 1} ({escape(symbol)})"
                    "<br>Offset: %{x:.6f} s<br>Amplitude: %{y:.6g}<extra></extra>"
                ),
            )
        )
    if products.morphology_samples.shape[0]:
        mean = np.mean(products.morphology_samples[:, channel], axis=0)
        figure.add_trace(
            go.Scatter(
                x=products.morphology_time_seconds,
                y=mean,
                mode="lines",
                line={"color": ORANGE, "width": 2.2},
                name="Arithmetic mean",
                zorder=10,
            )
        )
    channel_info = products.recording.header.channels[channel]
    figure.update_xaxes(title_text="Time from annotated beat (s)")
    figure.update_yaxes(
        title_text=f"{channel_info.name} ({channel_info.units})",
    )
    return figure


__all__ = ["morphology_figure", "rr_figure", "waveform_figure"]
