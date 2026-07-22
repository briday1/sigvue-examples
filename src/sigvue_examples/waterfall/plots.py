"""Pure Plotly figure builders for analyzed waterfall products."""

from html import escape

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sigvue.plugin import add_viewport_heatmap

from .models import WaterfallProducts


def waterfall_figure(
    products: WaterfallProducts,
    *,
    viewport: object,
    colormap: str,
    zmin: float,
    zmax: float,
    spectrum_ymin: float,
    spectrum_ymax: float,
    spectrum_style: object,
    show_colorbar: bool,
    render_width: int,
    render_height: int,
    aggregation: str,
    annotations: tuple[object, ...] = (),
    annotation_color: str = "#ffffff",
    annotation_width: float = 1.5,
    annotation_opacity: float = 0.8,
) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=(0.12, 0.88),
        vertical_spacing=0.04,
    )
    figure.add_trace(go.Scatter(
        x=products.frequency_mhz,
        y=products.spectrum_dbfs,
        mode=spectrum_style.mode,
        line=spectrum_style.line,
        marker=spectrum_style.plotly_marker,
        name="Average spectrum",
    ), row=1, col=1)
    add_viewport_heatmap(
        figure,
        viewport=viewport,
        x=products.frequency_mhz,
        y=products.time_edges_ms,
        z=products.waterfall_dbfs,
        zmin=zmin,
        zmax=zmax,
        colorscale=colormap,
        showscale=show_colorbar,
        colorbar={"title": "dBFS"},
        render_width=render_width,
        render_height=render_height,
        aggregation=aggregation,
        row=2,
        col=1,
    )
    annotation_x: list[float | None] = []
    annotation_y: list[float | None] = []
    annotation_hover_x: list[float] = []
    annotation_hover_y: list[float] = []
    annotation_hover_text: list[str] = []
    for annotation in annotations:
        y0 = annotation.start_seconds * 1e3
        y1 = y0 + (annotation.duration_seconds or 0.0) * 1e3
        if y1 < products.time_edges_ms[0] or y0 > products.time_edges_ms[-1]:
            continue
        x0 = (
            annotation.frequency_lower_hz / 1e6
            if annotation.frequency_lower_hz is not None
            else float(products.frequency_mhz[0])
        )
        x1 = (
            annotation.frequency_upper_hz / 1e6
            if annotation.frequency_upper_hz is not None
            else float(products.frequency_mhz[-1])
        )
        if y1 > y0:
            annotation_x.extend((x0, x1, x1, x0, x0, None))
            annotation_y.extend((y0, y0, y1, y1, y0, None))
        else:
            annotation_x.extend((x0, x1, None))
            annotation_y.extend((y0, y0, None))
        annotation_hover_x.append((x0 + x1) / 2)
        annotation_hover_y.append((y0 + y1) / 2)
        stop_seconds = annotation.start_seconds + (annotation.duration_seconds or 0.0)
        frequency = (
            f"{annotation.frequency_lower_hz / 1e6:.9g}–{annotation.frequency_upper_hz / 1e6:.9g} MHz"
            if annotation.frequency_lower_hz is not None and annotation.frequency_upper_hz is not None
            else "Full displayed frequency span"
        )
        details = [
            f"<b>{escape(annotation.label or 'Annotation')}</b>",
            f"Time: {annotation.start_seconds:.9g}–{stop_seconds:.9g} s",
            f"Duration: {(annotation.duration_seconds or 0.0):.9g} s",
            f"Frequency: {escape(frequency)}",
        ]
        if annotation.comment:
            details.append(escape(annotation.comment))
        annotation_hover_text.append("<br>".join(details))
    if annotation_x:
        figure.add_trace(go.Scattergl(
            x=annotation_x,
            y=annotation_y,
            mode="lines",
            line={"color": annotation_color, "width": annotation_width},
            opacity=annotation_opacity,
            name="Annotations",
            showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)
        figure.add_trace(go.Scattergl(
            x=annotation_hover_x,
            y=annotation_hover_y,
            mode="markers",
            marker={
                "color": annotation_color,
                "size": max(8.0, annotation_width * 4),
                "opacity": max(0.15, min(0.45, annotation_opacity)),
                "symbol": "square-open",
            },
            text=annotation_hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Annotation details",
            showlegend=False,
        ), row=2, col=1)
    figure.update_yaxes(
        title_text="Power (dBFS)", range=[spectrum_ymin, spectrum_ymax],
        autorange=False, row=1, col=1,
    )
    figure.update_yaxes(
        title_text="Recording time (ms)",
        range=[float(products.time_edges_ms[0]), float(products.time_edges_ms[-1])],
        autorange=False,
        row=2,
        col=1,
    )
    frequency_step = (
        float(abs(products.frequency_mhz[1] - products.frequency_mhz[0]))
        if products.frequency_mhz.size > 1 else 1.0
    )
    frequency_range = [
        float(products.frequency_mhz[0] - frequency_step / 2),
        float(products.frequency_mhz[-1] + frequency_step / 2),
    ]
    figure.update_xaxes(
        title_text="RF frequency (MHz)",
        range=frequency_range,
        autorange=False,
        row=2,
        col=1,
    )
    figure.update_layout(uirevision=f"lte-waterfall:{products.recording.metadata_path}")
    return figure
