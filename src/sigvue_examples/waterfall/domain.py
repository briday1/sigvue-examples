"""One windowed spectrum/waterfall pipeline for files and SigMF collections."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sigvue.plugin import (
    Annotation,
    AnnotationField,
    AnnotationRequest,
    Annotator,
    Delivery,
    DataResource,
    Source,
    DeliveryContext,
    DirectorySource,
    ParameterContext,
    RasterizedHeatmap,
    TraceStyle,
    ViewContext,
)

from ..io.sigmf.capabilities import (
    WaterfallSigMFAnnotator,
    read_sigmf_annotations,
    sigmf_discovery_summary,
)
from ..io.sigmf.recording import SigMFRecording, load_metadata, load_recording
from ..style import style_figure


COLORMAPS = ("Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo", "Blues", "Greens", "Hot", "Jet")

def _rgba(color: str, alpha: float) -> str:
    value = color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha:g})"


@dataclass(frozen=True)
class WaterfallWindow:
    """A selected sample interval consumed by the waterfall analyses."""

    recording: SigMFRecording | GroupedSigMFRecording
    start_sample: int
    samples: tuple[np.ndarray, ...]
    channel_start_samples: tuple[int, ...] = ()

    @property
    def sample_rate(self) -> float:
        return self.recording.sample_rate


@dataclass(frozen=True)
class GroupedSigMFRecording:
    """Several synchronized single-channel SigMF files presented as one recording."""

    recordings: tuple[SigMFRecording, ...]
    labels: tuple[str, ...]
    collection_path: Path

    @property
    def metadata_path(self) -> Path:
        return self.recordings[0].metadata_path

    @property
    def metadata(self) -> dict[str, object]:
        return self.recordings[0].metadata

    @property
    def sample_rate(self) -> float:
        return self.recordings[0].sample_rate

    @property
    def sample_count(self) -> int:
        return max(recording.sample_count for recording in self.recordings)

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate

    @property
    def member_durations_seconds(self) -> tuple[float, ...]:
        return tuple(recording.duration_seconds for recording in self.recordings)


class GroupedWaterfallSigMFAnnotator(Annotator[GroupedSigMFRecording, WaterfallWindow]):
    """Apply the shared waterfall annotation contract to the active collection member."""

    timeline_color_control = "waterfall_annotation_region_color"

    def __init__(self) -> None:
        self._member_annotator = WaterfallSigMFAnnotator(
            "waterfall-member",
            self.timeline_color_control,
        )

    @property
    def fields(self) -> tuple[AnnotationField, ...]:
        return self._member_annotator.fields

    def discover(self, recording: GroupedSigMFRecording) -> tuple[Annotation, ...]:
        discovered = []
        for index, (label, member) in enumerate(zip(recording.labels, recording.recordings)):
            for annotation in read_sigmf_annotations(member):
                discovered.append(replace(
                    annotation,
                    identifier=f"{index}:{annotation.identifier}",
                    label=f"{label} · {annotation.label or 'Annotation'}",
                    view_selections={"waterfall-member": index},
                ))
        return tuple(discovered)

    def annotate(
        self,
        recording: GroupedSigMFRecording,
        delivered: WaterfallWindow,
        request: AnnotationRequest,
    ) -> Annotation:
        selected = request.view_selections.get("waterfall-member", 0)
        if selected >= len(recording.recordings):
            raise ValueError("Selected waterfall member is not available")
        annotation = self._member_annotator.annotate(
            recording.recordings[selected],
            delivered,
            request,
        )
        return replace(annotation, view_selections={"waterfall-member": selected})


class SigMFCollectionSource(Source[GroupedSigMFRecording]):
    """Discover collection manifests and load every declared recording member."""

    def __init__(self, directory: Path, filename: str) -> None:
        self.directory = directory.expanduser().resolve()
        self.filename = filename

    def discover(self) -> list[DataResource]:
        resources = []
        for collection_path in sorted(self.directory.rglob(self.filename)):
            if not collection_path.is_file():
                continue
            payload = json.loads(collection_path.read_text(encoding="utf-8"))
            members = payload.get("members", ())
            if not members:
                continue
            first_metadata_path = collection_path.parent / str(members[0]["metadata"])
            first_metadata = load_metadata(first_metadata_path)
            resources.append(DataResource(
                identifier=collection_path.name.removesuffix(".sigmf-collection"),
                title=str(payload.get("collection", {}).get("name") or collection_path.stem),
                source=collection_path,
                subtitle=f"{len(members)} collection members",
                timestamp=datetime.fromtimestamp(collection_path.stat().st_mtime, tz=timezone.utc),
                tags=("sigmf", "collection", "multi-recording"),
                summary=sigmf_discovery_summary(first_metadata),
                navigation_path=(),
            ))
        return resources

    def open(self, resource: DataResource) -> GroupedSigMFRecording:
        collection_path = Path(resource.source)
        payload = json.loads(collection_path.read_text(encoding="utf-8"))
        role_order = {"calibration": 0, "terminated-noise": 1, "ota": 2}
        members = sorted(
            payload["members"],
            key=lambda member: (role_order.get(str(member["role"]), 99), int(member["channel"])),
        )
        paths = tuple(collection_path.parent / str(member["metadata"]) for member in members)
        recordings = tuple(load_recording(path) for path in paths)
        if not recordings:
            raise ValueError("A grouped recording requires at least one channel")
        if any(recording.channel_count != 1 for recording in recordings):
            raise ValueError("Grouped recording members must each contain one channel")
        if len({recording.sample_rate for recording in recordings}) != 1:
            raise ValueError("Grouped recording members must share one sample rate")
        role_labels = {
            "calibration": "Calibration",
            "terminated-noise": "Terminated noise",
            "ota": "OTA",
        }
        labels = tuple(
            f"{role_labels.get(str(member['role']), str(member['role']).replace('-', ' ').title())}"
            f" · Channel {int(member['channel'])}"
            for member in members
        )
        return GroupedSigMFRecording(recordings, labels, collection_path)


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
    time_coordinates_are_edges: bool = False,
) -> None:
    """Draw visible standard SigMF annotation bounds with hover-only descriptions."""
    if not show_annotations or frequency_mhz.size == 0:
        return
    time_coordinates = np.sort(np.asarray(waterfall_time_ms, dtype=float))
    view_start = (
        float(time_coordinates[0]) * 1e-3
        if time_coordinates_are_edges
        else data.start_sample / data.sample_rate
    )
    view_stop = (
        float(time_coordinates[-1]) * 1e-3
        if time_coordinates_are_edges
        else (data.start_sample + data.samples[0].size) / data.sample_rate
    )
    view_lower_hz = float(np.min(frequency_mhz)) * 1e6
    view_upper_hz = float(np.max(frequency_mhz)) * 1e6
    displayed_time_start_ms = view_start * 1e3
    displayed_time_stop_ms = view_stop * 1e3
    displayed_time_edges = (
        time_coordinates
        if time_coordinates_are_edges
        else _cell_edges(time_coordinates, displayed_time_start_ms, displayed_time_stop_ms)
    )
    displayed_bin_count = displayed_time_edges.size - 1
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
        first_bin = min(displayed_bin_count - 1, max(0, first_bin))
        last_bin = min(displayed_bin_count - 1, max(first_bin, last_bin))
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
    root = directory.expanduser().resolve()

    def describe(metadata_path: Path) -> DataResource:
        resource = _describe_recording(metadata_path)
        relative = metadata_path.relative_to(root).as_posix().removesuffix(".sigmf-meta")
        return replace(resource, identifier=relative.replace("/", "::"))

    return DirectorySource(
        root,
        pattern=filename,
        loader=load_recording,
        describe=describe,
        recursive=recursive,
    )


class WindowedWaterfallDelivery(
    Delivery[SigMFRecording | GroupedSigMFRecording, WaterfallWindow]
):
    """Select a short interval over sparse power samples from a large recording."""

    def prepare(
        self,
        recording: SigMFRecording | GroupedSigMFRecording,
        ui: DeliveryContext,
    ) -> WaterfallWindow:
        overview_key = (
            recording.collection_path
            if isinstance(recording, GroupedSigMFRecording)
            else recording.metadata_path
        )
        overviews = ui.once(
            f"waterfall-power-overviews:{overview_key}",
            lambda: _sparse_power_overviews(recording),
        )
        has_member_switcher = isinstance(recording, GroupedSigMFRecording) and len(overviews) > 1
        start_seconds, end_seconds = ui.windowed(
            duration=recording.duration_seconds,
            default_window=min(0.02, recording.duration_seconds),
            minimum_window=min(0.002, recording.duration_seconds),
            step=min(0.002, recording.duration_seconds),
            overview=None if has_member_switcher else overviews[0],
            overview_series=overviews if has_member_switcher else None,
            overview_durations=(
                recording.member_durations_seconds
                if has_member_switcher and isinstance(recording, GroupedSigMFRecording)
                else None
            ),
            overview_switcher="waterfall-member" if has_member_switcher else None,
            overview_label="Received power (dBFS)",
            time_unit="auto",
        )
        start = round(start_seconds * recording.sample_rate)
        count = max(1, round((end_seconds - start_seconds) * recording.sample_rate))
        if isinstance(recording, GroupedSigMFRecording):
            starts = tuple(
                min(
                    recording_member.sample_count - min(count, recording_member.sample_count),
                    start,
                )
                for recording_member in recording.recordings
            )
            stops = tuple(
                min(recording_member.sample_count, member_start + count)
                for recording_member, member_start in zip(recording.recordings, starts)
            )
            member_samples = tuple(
                recording_member.read(member_start, member_stop - member_start)[0]
                for recording_member, member_start, member_stop in zip(recording.recordings, starts, stops)
            )
            return WaterfallWindow(
                recording,
                start,
                member_samples,
                starts,
            )
        return WaterfallWindow(recording, start, tuple(recording.read(start, count)))


def _sparse_power_overviews(
    recording: SigMFRecording | GroupedSigMFRecording,
    bins: int = 400,
    samples_per_bin: int = 4096,
) -> tuple[np.ndarray, ...]:
    if isinstance(recording, GroupedSigMFRecording):
        overviews = []
        for member in recording.recordings:
            bin_count = min(bins, member.sample_count)
            starts = np.linspace(
                0,
                max(0, member.sample_count - samples_per_bin),
                bin_count,
                dtype=np.int64,
            )
            values = np.empty(bin_count)
            for index, start in enumerate(starts):
                samples = member.read(
                    int(start),
                    min(samples_per_bin, member.sample_count - int(start)),
                )[0]
                values[index] = 10 * np.log10(max(float(np.mean(np.abs(samples) ** 2)), 1e-12))
            overviews.append(values)
        return tuple(overviews)

    bin_count = min(bins, recording.sample_count)
    starts = np.linspace(0, max(0, recording.sample_count - samples_per_bin), bin_count, dtype=np.int64)
    channel_count = recording.channel_count
    values = np.empty((channel_count, bin_count))
    for index, start in enumerate(starts):
        samples = recording.read(
            int(start),
            min(samples_per_bin, recording.sample_count - int(start)),
        )
        channel_power = np.mean(np.abs(samples) ** 2, axis=1)
        values[:, index] = 10 * np.log10(np.maximum(channel_power, 1e-12))
    return tuple(values[index] for index in range(channel_count))


def _waterfall_spectrogram(
    samples: np.ndarray,
    fft_size: int,
    maximum_rows: int,
    fft_window: str = "Hann",
    overlap_percent: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the shared overlapped STFT used by every waterfall workspace."""
    if samples.size < fft_size:
        samples = np.pad(samples, (0, fft_size - samples.size))
    hop = max(1, round(fft_size * (1 - overlap_percent / 100)))
    available_starts = np.arange(0, samples.size - fft_size + 1, hop, dtype=np.int64)
    selected = np.linspace(
        0,
        available_starts.size - 1,
        min(maximum_rows, available_starts.size),
        dtype=np.int64,
    )
    starts = available_starts[selected]
    blocks = np.asarray([samples[start : start + fft_size] for start in starts])
    window = {
        "Hann": np.hanning,
        "Hamming": np.hamming,
        "Blackman": np.blackman,
        "Rectangular": np.ones,
    }[fft_window](fft_size)
    spectra = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    power = (np.abs(spectra) / max(np.sum(window), 1)) ** 2
    # Keep the useful deep sidelobes of deterministic signals.  A -180 dB
    # plotting floor hid real calibration structure around -200 dB.
    plotting_floor = 1e-30
    power_dbfs = 10 * np.log10(np.maximum(power, plotting_floor))
    average_dbfs = 10 * np.log10(np.maximum(np.mean(power, axis=0), plotting_floor))
    time_edges = _cell_edges(
        starts + fft_size / 2,
        float(starts[0]),
        float(starts[-1] + fft_size),
    )
    return power_dbfs, average_dbfs, time_edges


@dataclass(frozen=True)
class WaterfallSettings:
    fft_size: int
    fft_window: str
    overlap_percent: int
    maximum_rows: int


@dataclass(frozen=True)
class WaterfallChannelProducts:
    label: str
    data: WaterfallWindow
    waterfall_dbfs: np.ndarray
    average_dbfs: np.ndarray
    frequency_mhz: np.ndarray
    time_edges_ms: np.ndarray
    center_hz: float
    frequency_bounds_mhz: tuple[float, float]
    time_bounds_ms: tuple[float, float]


@dataclass(frozen=True)
class WaterfallProducts:
    channels: tuple[WaterfallChannelProducts, ...]
    settings: WaterfallSettings


def configure_waterfall(data: WaterfallWindow, ui: ParameterContext) -> WaterfallSettings:
    return WaterfallSettings(
        fft_size=int(ui.select(
            "waterfall_fft_size",
            label="FFT size (samples)",
            default=4096,
            options=(1024, 2048, 4096, 8192, 16384),
            group="Spectrogram processing",
        )),
        fft_window=str(ui.select(
            "waterfall_fft_window",
            label="Fast-time window",
            default="Hann",
            options=("Hann", "Hamming", "Blackman", "Rectangular"),
            group="Spectrogram processing",
        )),
        overlap_percent=int(ui.select(
            "waterfall_overlap_percent",
            label="Slow-time overlap (%)",
            default=50,
            options=(0, 25, 50, 75, 88),
            group="Spectrogram processing",
        )),
        maximum_rows=int(ui.number(
            "waterfall_maximum_time_bins",
            label="Maximum slow-time bins",
            default=200,
            minimum=25,
            maximum=500,
            step=25,
            group="Spectrogram processing",
        )),
    )


def process_waterfall(data: WaterfallWindow, settings: WaterfallSettings) -> WaterfallProducts:
    if isinstance(data.recording, GroupedSigMFRecording):
        recordings = data.recording.recordings
        labels = data.recording.labels
        starts = data.channel_start_samples
        channel_samples = data.samples
    else:
        recordings = (data.recording,) * len(data.samples)
        labels = tuple(f"Channel {index + 1}" for index in range(len(data.samples)))
        starts = (data.start_sample,) * len(data.samples)
        channel_samples = data.samples

    channels = []
    for recording, label, start, samples in zip(recordings, labels, starts, channel_samples):
        fft_size = min(settings.fft_size, max(8, samples.size))
        waterfall_dbfs, average_dbfs, row_time_edges_samples = _waterfall_spectrogram(
            samples,
            fft_size,
            settings.maximum_rows,
            settings.fft_window,
            settings.overlap_percent,
        )
        frequency_offset = np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / data.sample_rate))
        captures = recording.metadata.get("captures", [{}])
        center_hz = float(captures[0].get("core:frequency", 0.0)) if captures else 0.0
        frequency_mhz = (center_hz + frequency_offset) / 1e6
        channel_data = WaterfallWindow(recording, start, (samples,))
        channels.append(WaterfallChannelProducts(
            label=label,
            data=channel_data,
            waterfall_dbfs=waterfall_dbfs,
            average_dbfs=average_dbfs,
            frequency_mhz=frequency_mhz,
            time_edges_ms=(start + row_time_edges_samples) / data.sample_rate * 1e3,
            center_hz=center_hz,
            frequency_bounds_mhz=_axis_bounds(frequency_mhz),
            time_bounds_ms=(
                (start + row_time_edges_samples[0]) / data.sample_rate * 1e3,
                (start + row_time_edges_samples[-1]) / data.sample_rate * 1e3,
            ),
        ))
    return WaterfallProducts(tuple(channels), settings)


def _waterfall_channel_figure(
    products: WaterfallChannelProducts,
    *,
    theme: str,
    colormap: str,
    limits: tuple[float, float],
    annotation_style: TraceStyle,
    show_annotations: bool,
    scale_revision: str,
    render_width: int,
    render_height: int,
    aggregation: str,
) -> go.Figure:
    data = products.data
    zmin, zmax = limits
    frequency_mhz = products.frequency_mhz
    view_start_ms, view_stop_ms = products.time_bounds_ms
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.1, 0.9],
        vertical_spacing=0.04,
    )
    figure.add_trace(
        go.Scatter(
            x=frequency_mhz,
            y=products.average_dbfs,
            name="Average spectrum",
            line={"color": "#087e8b"},
        ),
        row=1,
        col=1,
    )
    RasterizedHeatmap.create(
        x=frequency_mhz,
        y=products.time_edges_ms,
        z=products.waterfall_dbfs,
        zmin=zmin,
        zmax=zmax,
        colorscale=colormap,
        colorbar={"title": "dBFS"},
        render_width=render_width,
        render_height=render_height,
        aggregation=aggregation,
    ).add_to(figure, row=2, col=1)
    figure.add_trace(
        go.Scatter(
            x=(float(frequency_mhz[0]), float(frequency_mhz[-1])),
            y=(float(products.time_edges_ms[0]), float(products.time_edges_ms[-1])),
            mode="markers",
            marker={"opacity": 0.0, "size": 1},
            hoverinfo="skip",
            showlegend=False,
            name="Selection surface",
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title_text="Power (dBFS)",
        range=[zmin, zmax],
        autorange=False,
        uirevision=(
            f"waterfall-power:{zmin:.9g}:{zmax:.9g}:{scale_revision}"
        ),
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Recording time (ms)",
        range=[view_start_ms, view_stop_ms],
        autorange=False,
        uirevision=f"waterfall-time:{view_start_ms:.12g}:{view_stop_ms:.12g}",
        row=2,
        col=1,
    )
    figure.update_xaxes(
        range=list(products.frequency_bounds_mhz),
        autorange=False,
        uirevision=(
            f"waterfall-frequency:{products.frequency_bounds_mhz[0]:.12g}:"
            f"{products.frequency_bounds_mhz[1]:.12g}"
        ),
    )
    figure.update_xaxes(title_text="RF frequency (MHz)", row=2, col=1)
    figure.update_layout(
        uirevision=(
            f"waterfall:{data.recording.metadata_path.name}:{products.label}:"
            f"annotations-{show_annotations}"
        ),
        datarevision=scale_revision,
    )
    _add_sigmf_annotation_regions(
        figure,
        data,
        frequency_mhz,
        products.time_edges_ms,
        annotation_style,
        show_annotations,
        row=2,
        col=1,
        time_coordinates_are_edges=True,
    )

    return style_figure(
        figure,
        theme,
        f"{products.label} · spectrum and waterfall",
    )


def _rounded_limits(lower: float, upper: float, *, minimum_span: float = 10.0) -> tuple[float, float]:
    lower = float(np.clip(lower, -300.0, 6.0))
    upper = float(np.clip(upper, -300.0, 6.0))
    if upper - lower < minimum_span:
        midpoint = (lower + upper) / 2
        lower, upper = midpoint - minimum_span / 2, midpoint + minimum_span / 2
        if lower < -300.0:
            lower, upper = -300.0, -300.0 + minimum_span
        elif upper > 6.0:
            lower, upper = 6.0 - minimum_span, 6.0
    return float(np.floor(lower)), float(np.ceil(upper))


def _automatic_dbfs_limits(products: WaterfallChannelProducts) -> tuple[float, float]:
    """Discover one shared spectrum/waterfall range whenever the view refreshes."""
    finite = products.waterfall_dbfs[np.isfinite(products.waterfall_dbfs)]
    if finite.size == 0:
        return -100.0, -20.0
    upper_candidates = [float(np.percentile(finite, 99.5)) + 3.0]
    finite_average = products.average_dbfs[np.isfinite(products.average_dbfs)]
    if finite_average.size:
        upper_candidates.append(float(np.max(finite_average)) + 3.0)
    upper = max(upper_candidates)
    median = float(np.median(finite))
    peak = upper - 3.0
    lower = median - 6.0 if peak - median > 60.0 else float(np.percentile(finite, 5.0)) - 3.0
    return _rounded_limits(lower, upper)


def _rendered_dimension(size: int, limit: int) -> int:
    block = (size + limit - 1) // limit
    return (size + block - 1) // block


def present_waterfall(products: WaterfallProducts, ui: ViewContext) -> None:
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
    with ui.details_group("Raster rendering"):
        render_width = int(ui.select(
            "waterfall_render_width",
            label="Heatmap render width",
            default=1024,
            options=(256, 512, 1024, 2048),
        ))
        render_height = int(ui.select(
            "waterfall_render_height",
            label="Heatmap render height",
            default=512,
            options=(128, 256, 512, 1024),
        ))
        aggregation = str(ui.select(
            "waterfall_render_aggregation",
            label="Heatmap aggregation",
            default="mean",
            options=("max", "mean", "median"),
        ))
    auto_scale = ui.toggle(
        "waterfall_auto_dbfs_scale",
        default=True,
        label="Auto-discover scales from each member",
        group="Spectrogram display",
    )
    zmin, zmax = ui.limits(
        "waterfall_dbfs_limits",
        label="Manual dBFS limits (Auto off)",
        default=(-90.0, -20.0),
        minimum=-300.0,
        maximum=6.0,
        step=1.0,
        group="Spectrogram display",
    )
    scale_revision = ":".join((
        str(products.settings.fft_size),
        products.settings.fft_window,
        str(products.settings.overlap_percent),
        str(products.settings.maximum_rows),
        str(auto_scale),
        f"{zmin:.9g}",
        f"{zmax:.9g}",
        colormap,
        str(render_width),
        str(render_height),
        aggregation,
        str(show_annotations),
        annotation_style.color,
        f"{annotation_style.width:.9g}",
        annotation_style.line_style,
        annotation_style.marker,
        f"{annotation_style.opacity:.9g}",
    ))
    figures = {
        channel.label: _waterfall_channel_figure(
            channel,
            theme=ui.theme,
            colormap=colormap,
            limits=_automatic_dbfs_limits(channel) if auto_scale else (zmin, zmax),
            annotation_style=annotation_style,
            show_annotations=show_annotations,
            scale_revision=scale_revision,
            render_width=render_width,
            render_height=render_height,
            aggregation=aggregation,
        )
        for channel in products.channels
    }
    first = products.channels[0]
    ui.stat("Center frequency", f"{first.center_hz / 1e6:g} MHz")
    ui.stat("Sample rate", f"{first.data.sample_rate / 1e6:g} MS/s")
    ui.stat("Members", len(products.channels))
    full_cells = sum(channel.waterfall_dbfs.size for channel in products.channels)
    rendered_cells = sum(
        _rendered_dimension(channel.waterfall_dbfs.shape[0], render_height)
        * _rendered_dimension(channel.waterfall_dbfs.shape[1], render_width)
        for channel in products.channels
    )
    ui.stat("Rendered heatmap pixels", f"{rendered_cells:,} from {full_cells:,} cells")
    with ui.tab("Spectrum + waterfall"):
        if len(figures) == 1:
            ui.plot(next(iter(figures.values())), key="waterfall-spectrum", axis_navigation="bounded")
        else:
            ui.view_switcher(
                "Recording member",
                figures,
                key="waterfall-member",
                selector="dropdown",
                axis_navigation="bounded",
            )
