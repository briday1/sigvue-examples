"""Static Matplotlib presentation of the shared waterfall analysis products."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from sigvue.plugin import Presentation, ViewContext

from .domain import COLORMAPS, WaterfallChannelProducts, WaterfallProducts


def _edges(centers: np.ndarray) -> np.ndarray:
    values = np.asarray(centers, dtype=float)
    if values.size == 1:
        return np.asarray((values[0] - 0.5, values[0] + 0.5))
    midpoints = (values[:-1] + values[1:]) / 2
    return np.concatenate(([values[0] - (midpoints[0] - values[0])], midpoints, [values[-1] + (values[-1] - midpoints[-1])]))


def waterfall_figure(
    products: WaterfallChannelProducts,
    theme: str,
    colormap: str,
    limits: tuple[float, float],
) -> Figure:
    dark = theme == "dark"
    background = "#10252d" if dark else "#ffffff"
    foreground = "#e7f1f3" if dark else "#13212b"
    grid = "#36515b" if dark else "#dce5e8"
    figure = Figure(figsize=(11, 7), facecolor=background, constrained_layout=True)
    grid_spec = figure.add_gridspec(2, 1, height_ratios=(1, 9))
    spectrum = figure.add_subplot(grid_spec[0])
    waterfall = figure.add_subplot(grid_spec[1], sharex=spectrum)
    spectrum.plot(products.frequency_mhz, products.average_dbfs, color="#55b9c3" if dark else "#087e8b", linewidth=1)
    spectrum.set_ylabel("dBFS")
    spectrum.set_ylim(*limits)
    image = waterfall.pcolormesh(
        _edges(products.frequency_mhz),
        products.time_edges_ms,
        products.waterfall_dbfs,
        cmap=colormap.lower(),
        vmin=limits[0],
        vmax=limits[1],
        shading="flat",
        rasterized=True,
    )
    waterfall.set_xlabel("RF frequency (MHz)")
    waterfall.set_ylabel("Recording time (ms)")
    waterfall.set_xlim(*products.frequency_bounds_mhz)
    waterfall.set_ylim(*products.time_bounds_ms)
    colorbar = figure.colorbar(image, ax=waterfall, pad=0.015)
    colorbar.set_label("dBFS", color=foreground)
    colorbar.ax.tick_params(colors=foreground)
    for axis in (spectrum, waterfall):
        axis.set_facecolor(background)
        axis.tick_params(colors=foreground)
        axis.xaxis.label.set_color(foreground)
        axis.yaxis.label.set_color(foreground)
        axis.grid(color=grid, linewidth=0.45, alpha=0.7)
        for spine in axis.spines.values():
            spine.set_color(grid)
    spectrum.tick_params(labelbottom=False)
    figure.suptitle(f"{products.label} · Matplotlib spectrum and waterfall", color=foreground, fontsize=14)
    return figure


class MatplotlibWaterfallPresentation(Presentation[WaterfallProducts]):
    def present(self, products: WaterfallProducts, ui: ViewContext) -> None:
        colormap = ui.colormap(
            "matplotlib_waterfall_colormap",
            label="Colormap",
            default="Plasma",
            options=COLORMAPS,
            group="Static waterfall display",
        )
        limits = ui.limits(
            "matplotlib_waterfall_dbfs_limits",
            label="Fixed dBFS limits",
            default=(-100.0, 0.0),
            minimum=-300.0,
            maximum=6.0,
            group="Static waterfall display",
        )
        figures = {
            channel.label: waterfall_figure(channel, ui.theme, colormap, limits)
            for channel in products.channels
        }
        first = products.channels[0]
        ui.stat("Renderer", "Matplotlib PNG")
        ui.stat("Center frequency", f"{first.center_hz / 1e6:g} MHz")
        ui.stat("Sample rate", f"{first.data.sample_rate / 1e6:g} MS/s")
        with ui.tab("Static spectrum + waterfall"):
            if len(figures) == 1:
                ui.plot(next(iter(figures.values())), key="matplotlib-waterfall")
            else:
                ui.view_switcher(
                    "Recording member",
                    figures,
                    key="matplotlib-waterfall-member",
                    selector="dropdown",
                )
