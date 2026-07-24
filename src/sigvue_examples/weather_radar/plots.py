"""Pure Plotly builders for exact and display-resampled radar views."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly import colors as plotly_colors

from ..plugins.nexrad import FIRST_MEASURED_CODE, NexradLevel3Radial
from ..style import COLORS, style_plotly
from .models import WeatherRadarProducts


NEXRAD_COLORSCALE = (
    (0.00, "#646464"),
    (0.15, "#04e9e7"),
    (0.28, "#019ff4"),
    (0.40, "#0300f4"),
    (0.48, "#02fd02"),
    (0.58, "#01c501"),
    (0.66, "#008e00"),
    (0.72, "#fdf802"),
    (0.78, "#e5bc00"),
    (0.84, "#fd9500"),
    (0.89, "#fd0000"),
    (0.94, "#d40000"),
    (0.97, "#bc0000"),
    (1.00, "#f800fd"),
)


def _register_nexrad_preview() -> None:
    """Make the exact custom scale available to Sigvue's visual picker."""
    sampled = plotly_colors.sample_colorscale(
        [list(stop) for stop in NEXRAD_COLORSCALE],
        [index / 100 for index in range(101)],
        colortype="rgb",
    )
    plotly_colors.sequential.NEXRAD = sampled


_register_nexrad_preview()

REFLECTIVITY_COLORMAPS = (
    "NEXRAD",
    "Turbo",
    "Viridis",
    "Cividis",
    "Plasma",
    "Inferno",
    "Magma",
    "Jet",
    "Rainbow",
    "Portland",
    "Hot",
)


def cartesian_display_grid(
    scan: NexradLevel3Radial,
    *,
    maximum_range_km: float,
    pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample exact polar gates at Cartesian pixel centers for display only."""
    if maximum_range_km <= 0 or pixels < 32:
        raise ValueError("display range and pixel count must be positive")
    edges = np.linspace(
        -maximum_range_km,
        maximum_range_km,
        pixels + 1,
        dtype=np.float64,
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    x, y = np.meshgrid(centers, centers)
    ground_range = np.hypot(x, y)
    slant_range = ground_range / scan.ground_range_scale
    azimuth = np.degrees(np.arctan2(x, y)) % 360.0

    order = np.argsort(scan.azimuth_start_deg)
    ordered_starts = scan.azimuth_start_deg[order]
    position = np.searchsorted(ordered_starts, azimuth, side="right") - 1
    position[position < 0] = len(order) - 1
    radial = order[position]
    angular_offset = (azimuth - scan.azimuth_start_deg[radial]) % 360.0
    covered = angular_offset <= scan.azimuth_width_deg[radial] + 1e-6

    gate = np.floor(
        (slant_range - scan.first_range_bin * scan.gate_size_km) / scan.gate_size_km
    ).astype(np.int32)
    valid = (
        covered
        & (ground_range <= maximum_range_km)
        & (gate >= 0)
        & (gate < scan.range_bin_count)
    )
    safe_gate = np.clip(gate, 0, scan.range_bin_count - 1)
    valid &= safe_gate < scan.radial_gate_counts[radial]
    codes = scan.level_codes[radial, safe_gate]
    measured = valid & (codes >= FIRST_MEASURED_CODE)
    dbz = np.full((pixels, pixels), np.nan, dtype=np.float32)
    dbz[measured] = (
        scan.header.minimum_value_dbz
        + (codes[measured].astype(np.float32) - FIRST_MEASURED_CODE)
        * scan.header.value_increment_dbz
    )
    return centers, centers, dbz


def ppi_figure(
    scan: NexradLevel3Radial,
    *,
    maximum_range_km: float,
    pixels: int,
    colormap: str,
    theme: str,
) -> go.Figure:
    if colormap not in REFLECTIVITY_COLORMAPS:
        raise ValueError(f"Unknown reflectivity colormap: {colormap}")
    x, y, dbz = cartesian_display_grid(
        scan,
        maximum_range_km=maximum_range_km,
        pixels=pixels,
    )
    figure = go.Figure(
        go.Heatmap(
            x=x,
            y=y,
            z=dbz,
            zmin=-20,
            zmax=75,
            colorscale=(NEXRAD_COLORSCALE if colormap == "NEXRAD" else colormap),
            colorbar={"title": "dBZ"},
            hovertemplate=(
                "East: %{x:.1f} km<br>North: %{y:.1f} km"
                "<br>Reflectivity: %{z:.1f} dBZ<extra></extra>"
            ),
            zsmooth=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=(scan.i_center_km,),
            y=(scan.j_center_km,),
            mode="markers",
            marker={
                "color": "white",
                "line": {"color": "black", "width": 1},
                "size": 8,
            },
            name=scan.header.radar_id,
            hovertemplate=f"{scan.header.radar_id}<extra></extra>",
        )
    )
    figure.update_xaxes(
        title_text="East of radar (km)",
        range=[-maximum_range_km, maximum_range_km],
        constrain="domain",
    )
    figure.update_yaxes(
        title_text="North of radar (km)",
        range=[-maximum_range_km, maximum_range_km],
        scaleanchor="x",
        scaleratio=1,
    )
    return style_plotly(
        figure,
        title=(
            f"{scan.header.radar_id} {scan.header.product_id} "
            f"base reflectivity · {scan.header.scan_time:%Y-%m-%d %H:%M:%S} UTC"
        ),
        theme=theme,
        boxed_axes=True,
    )


def histogram_figure(
    products: WeatherRadarProducts,
    theme: str,
) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=products.histogram_dbz,
            y=products.histogram_counts,
            width=products.scan.header.value_increment_dbz * 0.92,
            marker={"color": COLORS[1]},
            hovertemplate="%{x:.1f} dBZ<br>%{y:,} gates<extra></extra>",
            name="Native gates",
        )
    )
    figure.update_xaxes(title_text="Exact native reflectivity (dBZ)")
    figure.update_yaxes(title_text="Gate count")
    return style_plotly(
        figure,
        title="Measured-gate reflectivity distribution",
        theme=theme,
        boxed_axes=True,
    )


__all__ = [
    "NEXRAD_COLORSCALE",
    "REFLECTIVITY_COLORMAPS",
    "cartesian_display_grid",
    "histogram_figure",
    "ppi_figure",
]
