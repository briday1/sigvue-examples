"""Quiet Plotly defaults shared by the minimal examples."""

from __future__ import annotations

from typing import Any


COLORS = ("#087e8b", "#d35d35", "#7656a5", "#5b7f3b")


def style_figure(figure: Any, ui: Any, title: str) -> Any:
    dark = ui.theme == "dark"
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
    figure.update_xaxes(showline=True, mirror=True, linecolor=grid, gridcolor=grid, zeroline=False)
    figure.update_yaxes(showline=True, mirror=True, linecolor=grid, gridcolor=grid, zeroline=False)
    return figure
