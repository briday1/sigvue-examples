"""Windowed spectrum/waterfall workspaces for LTE and radio astronomy data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sigvue.plugin import AnalysisContext, AnalysisWorkspace, DataDelivery, DataResource, DirectorySource, TraceStyle

from .capabilities import (
    SIGNAL_DISCOVERY_COLUMNS,
    SigMFExporter,
    WaterfallSigMFAnnotator,
    read_sigmf_annotations,
    sigmf_discovery_summary,
)
from .sigmf import SigMFRecording, load_metadata, load_recording
from .style import style_figure


COLORMAPS = ("Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo", "Blues", "Greens", "Hot", "Jet")
FFT_SIZES = (512, 1024, 2048, 4096, 8192, 16384)
FFT_WINDOWS = ("Hann", "Hamming", "Blackman", "Rectangular")
OVERLAPS = (0, 25, 50, 75, 88)
DBFS_MIN = -90.0
DBFS_MAX = -20.0


def _rgba(color: str, alpha: float) -> str:
    value = color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha:g})"


@dataclass(frozen=True)
class WaterfallWindow:
    """A selected sample interval consumed by the waterfall analyses."""

    recording: SigMFRecording
    start_sample: int
    samples: np.ndarray

    @property
    def sample_rate(self) -> float:
        return self.recording.sample_rate


def _axis_bounds(values: np.ndarray) -> tuple[float, float]:
    """Return the outer cell edges for an ordered Plotly heatmap coordinate."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size < 2:
        center = float(ordered[0])
        return center - 0.5, center + 0.5
    spacing = float(np.median(np.diff(ordered)))
    return float(ordered[0] - spacing / 2), float(ordered[-1] + spacing / 2)


def _cell_edges(centers: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Map displayed row centers to contiguous cells spanning the delivered view."""
    ordered = np.sort(np.asarray(centers, dtype=float))
    if ordered.size == 1:
        return np.asarray([lower, upper], dtype=float)
    return np.concatenate(([lower], (ordered[:-1] + ordered[1:]) / 2, [upper]))


def _add_sigmf_annotation_regions(
    figure: go.Figure,
    data: WaterfallWindow,
    frequency_mhz: np.ndarray,
    waterfall_time_ms: np.ndarray,
    annotation_style: TraceStyle,
    show_annotations: bool,
    *,
    row: int,
    col: int,
) -> None:
    """Draw visible standard SigMF annotation bounds with hover-only descriptions."""
    if not show_annotations or frequency_mhz.size == 0:
        return
    view_start = data.start_sample / data.sample_rate
    view_stop = (data.start_sample + data.samples.shape[-1]) / data.sample_rate
    view_lower_hz = float(np.min(frequency_mhz)) * 1e6
    view_upper_hz = float(np.max(frequency_mhz)) * 1e6
    displayed_times = np.sort(np.asarray(waterfall_time_ms, dtype=float))
    displayed_time_start_ms = view_start * 1e3
    displayed_time_stop_ms = view_stop * 1e3
    displayed_time_edges = _cell_edges(
        displayed_times,
        displayed_time_start_ms,
        displayed_time_stop_ms,
    )
    polygon_x: list[float | None] = []
    polygon_y: list[float | None] = []
    hover_x: list[float] = []
    hover_y: list[float] = []
    hover_text: list[str] = []
    for annotation in read_sigmf_annotations(data.recording):
        annotation_stop = (
            data.recording.duration_seconds
            if annotation.duration_seconds is None
            else annotation.start_seconds + annotation.duration_seconds
        )
        lower_hz = annotation.frequency_lower_hz if annotation.frequency_lower_hz is not None else view_lower_hz
        upper_hz = annotation.frequency_upper_hz if annotation.frequency_upper_hz is not None else view_upper_hz
        if annotation_stop < view_start or annotation.start_seconds > view_stop:
            continue
        if upper_hz < view_lower_hz or lower_hz > view_upper_hz:
            continue
        start_seconds = max(view_start, annotation.start_seconds)
        stop_seconds = min(view_stop, annotation_stop)
        visible_lower_hz = max(view_lower_hz, lower_hz)
        visible_upper_hz = min(view_upper_hz, upper_hz)
        description = annotation.comment or annotation.label or "Annotation"
        hover = (
            f"{description}<br>Time: {annotation.start_seconds:.9g}–{annotation_stop:.9g} s"
            f"<br>Frequency: {lower_hz:.12g}–{upper_hz:.12g} Hz"
        )
        exact_start_ms = start_seconds * 1e3
        exact_stop_ms = stop_seconds * 1e3
        first_bin = int(np.searchsorted(displayed_time_edges, exact_start_ms, side="right") - 1)
        last_bin = int(np.searchsorted(displayed_time_edges, exact_stop_ms, side="left") - 1)
        first_bin = min(displayed_times.size - 1, max(0, first_bin))
        last_bin = min(displayed_times.size - 1, max(first_bin, last_bin))
        visual_start_ms = min(exact_start_ms, float(displayed_time_edges[first_bin]))
        visual_stop_ms = max(exact_stop_ms, float(displayed_time_edges[last_bin + 1]))
        x = [visible_lower_hz / 1e6, visible_upper_hz / 1e6] * 2
        x = [x[0], x[1], x[1], x[0], x[0]]
        y = [visual_start_ms, visual_start_ms, visual_stop_ms, visual_stop_ms, visual_start_ms]
        polygon_x.extend((*x, None))
        polygon_y.extend((*y, None))
        hover_x.extend((x[0], (x[0] + x[1]) / 2, x[1]))
        hover_y.extend(((y[0] + y[2]) / 2,) * 3)
        hover_text.extend((hover,) * 3)
    if polygon_x:
        figure.add_trace(
            go.Scatter(
                x=polygon_x,
                y=polygon_y,
                mode="lines",
                line=annotation_style.line,
                fill="toself",
                fillcolor=_rgba(annotation_style.color, 0.12),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        figure.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                marker={"color": annotation_style.color, "opacity": 0.01, "size": 12},
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name="",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def _describe_recording(metadata_path: Path) -> DataResource:
    metadata = load_metadata(metadata_path)
    global_metadata = metadata["global"]
    channels = int(global_metadata.get("core:num_channels", 1))
    sample_rate = global_metadata.get("core:sample_rate")
    subtitle = f"{channels} channel{'s' if channels != 1 else ''}"
    if sample_rate is not None:
        subtitle += f" · {float(sample_rate):g} samples/s"
    return DataResource(
        identifier=metadata_path.name.removesuffix(".sigmf-meta"),
        title=str(global_metadata.get("core:description") or metadata_path.stem),
        source=metadata_path,
        subtitle=subtitle,
        timestamp=datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=timezone.utc),
        tags=("sigmf", str(global_metadata["core:datatype"])),
        summary=sigmf_discovery_summary(metadata),
    )


def _recording_source(directory: Path, filename: str, *, recursive: bool = False) -> DirectorySource:
    """Bind SigMF I/O to the browser contract inside this domain module."""
    return DirectorySource(
        directory,
        pattern=filename,
        loader=load_recording,
        describe=_describe_recording,
        recursive=recursive,
    )


class WindowedLteDelivery(DataDelivery[SigMFRecording, WaterfallWindow]):
    """Select an interval over a sliding-median power overview."""

    def prepare(self, recording: SigMFRecording, ui: AnalysisContext) -> WaterfallWindow:
        overview = ui.once(
            f"lte-median-power-overview:{recording.metadata_path}",
            lambda: _median_power_overview(recording),
        )
        start_seconds, end_seconds = ui.windowed(
            duration=recording.duration_seconds,
            default_window=min(0.005, recording.duration_seconds),
            minimum_window=min(0.001, recording.duration_seconds),
            step=min(0.001, recording.duration_seconds),
            overview=overview,
            overview_label="Sliding median power (dBFS)",
            time_unit="ms",
        )
        start = round(start_seconds * recording.sample_rate)
        count = max(1, round((end_seconds - start_seconds) * recording.sample_rate))
        return WaterfallWindow(recording, start, recording.read(start, count))


def _median_power_overview(recording: SigMFRecording) -> np.ndarray:
    bin_count = min(400, recording.sample_count)
    edges = np.linspace(0, recording.sample_count, bin_count + 1, dtype=np.int64)
    power = np.empty(bin_count)
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        samples = recording.read(int(start), int(end - start))[0]
        power[index] = np.mean(np.abs(samples) ** 2)
    power_dbfs = 10 * np.log10(np.maximum(power, 1e-12))
    width = min(9, bin_count)
    if width % 2 == 0:
        width -= 1
    padded = np.pad(power_dbfs, width // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, width), axis=1)


def analyze_lte(data: WaterfallWindow, ui: AnalysisContext) -> None:
    show_annotations = ui.toggle(
        "lte_show_annotations", default=True, label="Show annotations", group="Annotation display"
    )
    annotation_style = ui.trace_style(
        "lte_annotation_region",
        label="Annotation boxes",
        color="#ffffff",
        width=0.5,
        opacity=0.6,
        line_style="solid",
        group="Annotation display",
    )
    colormap = ui.colormap(
        "lte_colormap",
        label="Colormap",
        default="Plasma",
        options=COLORMAPS,
        group="Spectrogram display",
    )
    dbfs_min, dbfs_max = ui.limits(
        "lte_dbfs_limits",
        label="dBFS scale",
        default=(DBFS_MIN, DBFS_MAX),
        minimum=-120.0,
        maximum=0.0,
        step=1.0,
        group="Spectrogram display",
    )
    requested_fft_size = ui.select(
        "lte_fft_size",
        label="Fast-time FFT size (samples)",
        default=4096,
        options=FFT_SIZES,
        group="Spectrogram processing",
    )
    fft_window = ui.select(
        "lte_fft_window",
        label="Fast-time window",
        default="Hann",
        options=FFT_WINDOWS,
        group="Spectrogram processing",
    )
    overlap_percent = ui.select(
        "lte_overlap_percent",
        label="Slow-time overlap (%)",
        default=50,
        options=OVERLAPS,
        group="Spectrogram processing",
    )
    maximum_time_bins = ui.number(
        "lte_maximum_time_bins",
        label="Maximum slow-time bins",
        default=200,
        minimum=25,
        maximum=500,
        step=25,
        group="Spectrogram processing",
    )
    samples = data.samples[0]
    fft_size = min(int(requested_fft_size), samples.size)
    hop = max(1, round(fft_size * (1 - int(overlap_percent) / 100)))
    available_starts = np.arange(0, max(1, samples.size - fft_size + 1), hop)
    starts = available_starts[
        np.linspace(0, available_starts.size - 1, min(int(maximum_time_bins), available_starts.size), dtype=int)
    ]
    rows = np.asarray([samples[start : start + fft_size] for start in starts])
    taper = {
        "Hann": np.hanning,
        "Hamming": np.hamming,
        "Blackman": np.blackman,
        "Rectangular": np.ones,
    }[str(fft_window)](fft_size)
    spectra = np.fft.fftshift(np.fft.fft(rows * taper, axis=1), axes=1)
    power = np.abs(spectra / max(np.sum(taper), 1)) ** 2
    waterfall_db = 10 * np.log10(np.maximum(power, 1e-12))
    average_db = 10 * np.log10(np.maximum(np.mean(power, axis=0), 1e-12))

    captures = data.recording.metadata.get("captures", [])
    center_hz = float(captures[0].get("core:frequency", 0.0)) if captures else 0.0
    frequency_mhz = (center_hz + np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / data.sample_rate))) / 1e6
    time_ms = (data.start_sample + starts + fft_size / 2) / data.sample_rate * 1e3
    frequency_bounds_mhz = _axis_bounds(frequency_mhz)
    time_bounds_ms = (
        data.start_sample / data.sample_rate * 1e3,
        (data.start_sample + samples.size) / data.sample_rate * 1e3,
    )

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=(0.1, 0.9),
        vertical_spacing=0.04,
    )
    figure.add_trace(
        go.Scatter(x=frequency_mhz, y=average_db, mode="lines", line={"color": "#087e8b"}, showlegend=False),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            x=frequency_mhz,
            y=time_ms,
            z=waterfall_db,
            zmin=dbfs_min,
            zmax=dbfs_max,
            colorscale=colormap,
            colorbar={"title": "dBFS", "tickformat": ".1f"},
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title_text="PSD (dBFS)",
        range=[dbfs_min, dbfs_max],
        autorange=False,
        minallowed=dbfs_min,
        maxallowed=dbfs_max,
        autorangeoptions={"clipmin": dbfs_min, "clipmax": dbfs_max},
        tickformat=".1f",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Recording time (ms)",
        tickformat="07.2f",
        range=list(time_bounds_ms),
        autorange=False,
        minallowed=time_bounds_ms[0],
        maxallowed=time_bounds_ms[1],
        autorangeoptions={"clipmin": time_bounds_ms[0], "clipmax": time_bounds_ms[1]},
        row=2,
        col=1,
    )
    figure.update_xaxes(
        range=list(frequency_bounds_mhz),
        autorange=False,
        minallowed=frequency_bounds_mhz[0],
        maxallowed=frequency_bounds_mhz[1],
        autorangeoptions={"clipmin": frequency_bounds_mhz[0], "clipmax": frequency_bounds_mhz[1]},
    )
    figure.update_xaxes(title_text="RF frequency (MHz)", tickformat="07.2f", row=2, col=1)
    figure.update_layout(
        uirevision=f"lte-spectrum:{data.recording.metadata_path.name}:annotations-{show_annotations}"
    )
    _add_sigmf_annotation_regions(
        figure,
        data,
        frequency_mhz,
        time_ms,
        annotation_style,
        show_annotations,
        row=2,
        col=1,
    )

    ui.stat("Center frequency", f"{center_hz / 1e6:g} MHz")
    ui.stat("Sample rate", f"{data.sample_rate / 1e6:g} MS/s")
    ui.stat("Displayed samples", f"{samples.size:,}")
    with ui.tab("Spectrum + waterfall"):
        ui.plot(style_figure(figure, ui.theme, "LTE spectrum"), key="lte-spectrum")


def create_lte_workspace(config=None):
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data"))
    filename = str(values.get("filename", "*.sigmf-meta"))
    return AnalysisWorkspace(
        identifier="lte-waterfall",
        name="LTE Waterfall",
        description="Windowed mode: drag or resize an interval over sliding-median power and inspect its LTE time-frequency plot.",
        source=_recording_source(root, filename, recursive=True),
        delivery=WindowedLteDelivery(),
        annotator=WaterfallSigMFAnnotator("lte-spectrum", "lte_annotation_region_color"),
        exporter=SigMFExporter(),
        analyze=analyze_lte,
        category="spectrum monitoring",
        tags=("windowed", "single-channel", "LTE", "spectrogram", "waterfall"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )


class WindowedWaterfallDelivery(DataDelivery[SigMFRecording, WaterfallWindow]):
    """Select a short interval over sparse power samples from a large recording."""

    def prepare(self, recording: SigMFRecording, ui: AnalysisContext) -> WaterfallWindow:
        overview = ui.once(
            f"waterfall-power-overview:{recording.metadata_path}",
            lambda: _sparse_power_overview(recording),
        )
        start_seconds, end_seconds = ui.windowed(
            duration=recording.duration_seconds,
            default_window=min(0.02, recording.duration_seconds),
            minimum_window=min(0.002, recording.duration_seconds),
            step=min(0.002, recording.duration_seconds),
            overview=overview,
            overview_label="Sampled wideband power (dBFS)",
            time_unit="auto",
        )
        start = round(start_seconds * recording.sample_rate)
        count = max(1, round((end_seconds - start_seconds) * recording.sample_rate))
        return WaterfallWindow(recording, start, recording.read(start, count))


def _sparse_power_overview(
    recording: SigMFRecording,
    bins: int = 400,
    samples_per_bin: int = 4096,
) -> np.ndarray:
    bin_count = min(bins, recording.sample_count)
    starts = np.linspace(0, max(0, recording.sample_count - samples_per_bin), bin_count, dtype=np.int64)
    values = np.empty(bin_count)
    for index, start in enumerate(starts):
        samples = recording.read(int(start), min(samples_per_bin, recording.sample_count - int(start)))[0]
        values[index] = 10 * np.log10(max(float(np.mean(np.abs(samples) ** 2)), 1e-12))
    return values


def _waterfall_spectrogram(
    samples: np.ndarray,
    fft_size: int,
    maximum_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_count = samples.size // fft_size
    if block_count < 1:
        padded = np.pad(samples, (0, fft_size - samples.size))
        blocks = padded.reshape(1, fft_size)
        centers = np.asarray([samples.size / 2], dtype=float)
    else:
        selected = np.linspace(
            0,
            block_count - 1,
            min(maximum_rows, block_count),
            dtype=np.int64,
        )
        blocks = samples[: block_count * fft_size].reshape(block_count, fft_size)[selected]
        centers = (selected.astype(float) + 0.5) * fft_size
    window = np.hanning(fft_size)
    spectra = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    power = (np.abs(spectra) / max(np.sum(window), 1)) ** 2
    power_dbfs = 10 * np.log10(np.maximum(power, 1e-18))
    average_dbfs = 10 * np.log10(np.maximum(np.mean(power, axis=0), 1e-18))
    return power_dbfs, average_dbfs, centers


def analyze_waterfall(data: WaterfallWindow, ui: AnalysisContext) -> None:
    show_annotations = ui.toggle(
        "waterfall_show_annotations", default=True, label="Show annotations", group="Annotation display"
    )
    annotation_style = ui.trace_style(
        "waterfall_annotation_region",
        label="Annotation boxes",
        color="#ffffff",
        width=0.5,
        opacity=0.6,
        line_style="solid",
        group="Annotation display",
    )
    colormap = ui.colormap(
        "waterfall_colormap",
        label="Colormap",
        default="Plasma",
        options=COLORMAPS,
        group="Spectrogram display",
    )
    zmin, zmax = ui.limits(
        "waterfall_dbfs_limits",
        label="dBFS scale",
        default=(-100.0, -20.0),
        minimum=-140.0,
        maximum=0.0,
        step=1.0,
        group="Spectrogram display",
    )
    requested_fft = int(ui.select(
        "waterfall_fft_size",
        label="FFT size (samples)",
        default=4096,
        options=(1024, 2048, 4096, 8192, 16384),
        group="Spectrogram processing",
    ))
    maximum_rows = int(ui.number(
        "waterfall_maximum_time_bins",
        label="Maximum time bins",
        default=200,
        minimum=25,
        maximum=500,
        step=25,
        group="Spectrogram processing",
    ))

    samples = data.samples[0]
    fft_size = min(requested_fft, max(8, samples.size))
    waterfall_dbfs, average_dbfs, row_center_samples = _waterfall_spectrogram(
        samples, fft_size, maximum_rows
    )
    frequency_offset = np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / data.sample_rate))
    captures = data.recording.metadata.get("captures", [{}])
    center_hz = float(captures[0].get("core:frequency", 0.0)) if captures else 0.0
    frequency_mhz = (center_hz + frequency_offset) / 1e6
    frequency_bounds_mhz = _axis_bounds(frequency_mhz)
    view_start_ms = data.start_sample / data.sample_rate * 1e3
    view_stop_ms = (data.start_sample + data.samples.shape[1]) / data.sample_rate * 1e3
    recording_time_ms = (data.start_sample + row_center_samples) / data.sample_rate * 1e3

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7], vertical_spacing=0.06)
    figure.add_trace(
        go.Scatter(x=frequency_mhz, y=average_dbfs, name="Average spectrum", line={"color": "#087e8b"}),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            x=frequency_mhz,
            y=recording_time_ms,
            z=waterfall_dbfs,
            zmin=zmin,
            zmax=zmax,
            colorscale=colormap,
            colorbar={"title": "dBFS"},
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title_text="Power (dBFS)",
        range=[zmin, zmax],
        autorange=False,
        minallowed=zmin,
        maxallowed=zmax,
        autorangeoptions={"clipmin": zmin, "clipmax": zmax},
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Recording time (ms)",
        range=[view_start_ms, view_stop_ms],
        autorange=False,
        minallowed=view_start_ms,
        maxallowed=view_stop_ms,
        autorangeoptions={"clipmin": view_start_ms, "clipmax": view_stop_ms},
        row=2,
        col=1,
    )
    figure.update_xaxes(
        range=list(frequency_bounds_mhz),
        autorange=False,
        minallowed=frequency_bounds_mhz[0],
        maxallowed=frequency_bounds_mhz[1],
        autorangeoptions={"clipmin": frequency_bounds_mhz[0], "clipmax": frequency_bounds_mhz[1]},
    )
    figure.update_xaxes(title_text="RF frequency (MHz)", row=2, col=1)
    figure.update_layout(
        uirevision=f"waterfall:{data.recording.metadata_path.name}:annotations-{show_annotations}"
    )
    _add_sigmf_annotation_regions(
        figure,
        data,
        frequency_mhz,
        recording_time_ms,
        annotation_style,
        show_annotations,
        row=2,
        col=1,
    )

    ui.stat("Center frequency", f"{center_hz / 1e6:g} MHz")
    ui.stat("Sample rate", f"{data.sample_rate / 1e6:g} MS/s")
    ui.stat("Selected duration", f"{samples.size / data.sample_rate * 1e3:g} ms")
    with ui.tab("Spectrum + waterfall"):
        ui.plot(style_figure(figure, ui.theme, "Spectrum and waterfall"), key="waterfall-spectrum")


def create_workspace(config=None):
    """Create one reusable SigMF waterfall analysis instance."""
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data"))
    filename = str(values.get("filename", "*.sigmf-meta"))
    return AnalysisWorkspace(
        identifier="waterfall",
        name="Waterfall",
        description="Windowed mode: inspect a SigMF recording as an average spectrum and waterfall.",
        source=_recording_source(root, filename, recursive=True),
        delivery=WindowedWaterfallDelivery(),
        annotator=WaterfallSigMFAnnotator("waterfall-spectrum", "waterfall_annotation_region_color"),
        exporter=SigMFExporter(),
        analyze=analyze_waterfall,
        category="spectrum monitoring",
        tags=("windowed", "sigmf", "spectrogram", "waterfall"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )
