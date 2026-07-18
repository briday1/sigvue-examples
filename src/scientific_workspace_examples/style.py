"""Quiet Plotly defaults shared by the minimal examples."""

from __future__ import annotations

from colorsys import hsv_to_rgb
from typing import Any


COLORS = ("#087e8b", "#d35d35", "#7656a5", "#5b7f3b")
INK = "#13212b"
MUTED = "#60717d"
GRID = "#dce5e8"
TEAL = COLORS[0]
ORANGE = COLORS[1]


def style_figure(figure: Any, theme: str, title: str) -> Any:
    """Apply shared Plotly presentation without depending on browser context types."""
    dark = theme == "dark"
    grid = "#36515b" if dark else "#dce5e8"
    figure.update_layout(
        template="plotly_dark" if dark else "simple_white",
        paper_bgcolor="#10252d" if dark else "white",
        plot_bgcolor="#10252d" if dark else "white",
        title={"text": title, "x": 0.01, "y": 0.98, "xanchor": "left", "yanchor": "top", "font": {"size": 15}},
        margin={"l": 70, "r": 30, "t": 68, "b": 56},
        legend={
            "orientation": "h",
            "x": 0.99,
            "y": 0.98,
            "xanchor": "right",
            "yanchor": "top",
            "bgcolor": "rgba(16,37,45,0.72)" if dark else "rgba(255,255,255,0.82)",
        },
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor=grid,
        gridwidth=0.5,
        showline=True,
        mirror=True,
        linecolor=grid,
        zeroline=False,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=grid,
        gridwidth=0.5,
        showline=True,
        mirror=True,
        linecolor=grid,
        zeroline=False,
    )
    return figure


def hsv_channel_colors(count: int, *, saturation: float = 0.68, value: float = 0.78) -> tuple[str, ...]:
    """Return stable channel colors sampled uniformly around the HSV hue wheel."""
    if count < 1:
        return ()
    colors = []
    for index in range(count):
        red, green, blue = hsv_to_rgb(index / count, saturation, value)
        colors.append(f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}")
    return tuple(colors)


def style_plotly(
    figure: Any,
    *,
    title: str | None = None,
    theme: str = "light",
    boxed_axes: bool = False,
) -> Any:
    """Give Plotly figures a compact, consistent scientific plotting treatment."""
    figure.update_layout(
        template="plotly_dark" if theme == "dark" else "simple_white",
        paper_bgcolor="#10252d" if theme == "dark" else "white",
        plot_bgcolor="#10252d" if theme == "dark" else "white",
        font={"family": "system-ui, -apple-system, sans-serif", "color": "#e7f1f3" if theme == "dark" else INK, "size": 12},
        margin={"l": 62, "r": 28, "t": 52, "b": 54},
        hovermode="x unified",
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 15}},
        legend={"orientation": "h", "x": 0, "y": 1.12, "yanchor": "bottom"},
    )
    grid = "#36515b" if theme == "dark" else GRID
    muted = "#a9bdc2" if theme == "dark" else MUTED
    figure.update_xaxes(
        showgrid=True,
        showline=True,
        mirror=boxed_axes,
        linecolor=grid,
        gridcolor=grid,
        gridwidth=0.5,
        zeroline=False,
        ticks="outside",
        tickcolor=muted,
    )
    figure.update_yaxes(
        showgrid=True,
        showline=True,
        mirror=boxed_axes,
        linecolor=grid,
        gridcolor=grid,
        gridwidth=0.5,
        zeroline=False,
        ticks="outside",
        tickcolor=muted,
    )
    return figure
