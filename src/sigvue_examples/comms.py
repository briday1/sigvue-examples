"""Windowed constellation and eye-diagram workspace for digital communications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from sigvue.plugin import AnalysisContext, AnalysisWorkspace, DataDelivery, DataResource, DirectorySource

from .capabilities import SIGNAL_DISCOVERY_COLUMNS, SigMFAnnotator, SigMFExporter, sigmf_discovery_summary
from .sigmf import SigMFRecording, load_metadata, load_recording
from .style import COLORS, style_figure


@dataclass(frozen=True)
class CommsWindow:
    """The exact recording interval selected by this workspace's delivery policy."""

    recording: SigMFRecording
    start_sample: int
    samples: np.ndarray

    @property
    def sample_rate(self) -> float:
        return self.recording.sample_rate

    @property
    def time_seconds(self) -> np.ndarray:
        return (self.start_sample + np.arange(self.samples.shape[1])) / self.sample_rate

    @property
    def sample_positions(self) -> np.ndarray:
        return self.start_sample + np.arange(self.samples.shape[1])

    @property
    def uses_sample_coordinates(self) -> bool:
        return self.recording.metadata["global"].get("core:sample_rate") is None


class WindowedCommsDelivery(DataDelivery[SigMFRecording, CommsWindow]):
    """Select a short interval over a decimated received-power overview."""

    def prepare(self, recording: SigMFRecording, ui: AnalysisContext) -> CommsWindow:
        overview = ui.once(
            f"comms-power-overview:{recording.metadata_path}",
            lambda: _power_overview(recording),
        )
        normalized = recording.metadata["global"].get("core:sample_rate") is None
        duration = float(recording.sample_count) if normalized else recording.duration_seconds
        default_window = min(4096.0, duration) if normalized else min(0.03, duration)
        minimum_window = min(256.0, duration) if normalized else min(0.005, duration)
        step = min(128.0, duration) if normalized else min(0.005, duration)
        start_coordinate, end_coordinate = ui.windowed(
            duration=duration,
            default_window=default_window,
            minimum_window=minimum_window,
            step=step,
            overview=overview,
            overview_label="Received power",
            time_unit="samples" if normalized else "ms",
        )
        scale = 1.0 if normalized else recording.sample_rate
        start = round(start_coordinate * scale)
        count = max(1, round((end_coordinate - start_coordinate) * scale))
        return CommsWindow(recording, start, recording.read(start, count))


def _power_overview(recording: SigMFRecording) -> np.ndarray:
    samples = recording.read(0, recording.sample_count)[0]
    block_count = min(200, samples.size)
    block_size = max(1, samples.size // block_count)
    blocks = samples[: block_count * block_size].reshape(block_count, block_size)
    return 10 * np.log10(np.maximum(np.mean(np.abs(blocks) ** 2, axis=1), 1e-12))


def analyze(data: CommsWindow, ui: AnalysisContext) -> None:
    samples = data.samples[0]
    metadata = data.recording.metadata["global"]
    modulation = _modulation_label(metadata, data.recording.metadata_path.stem)
    if data.uses_sample_coordinates:
        samples_per_symbol = max(1, int(metadata.get("examples:samples_per_symbol", 8)))
        carrier_cycles_per_sample = float(metadata.get("examples:carrier_cycles_per_sample", 0.0))
        baseband = samples * np.exp(-1j * 2 * np.pi * carrier_cycles_per_sample * data.sample_positions)
        symbol_rate = None
    else:
        symbol_rate = float(metadata["examples:symbol_rate"])
        carrier_hz = float(metadata["examples:carrier_hz"])
        baseband = samples * np.exp(-1j * 2 * np.pi * carrier_hz * data.time_seconds)
        samples_per_symbol = max(1, round(data.sample_rate / symbol_rate))
    alignment = (-data.start_sample) % samples_per_symbol
    aligned = baseband[alignment:]

    symbol_count = aligned.size // samples_per_symbol
    symbols = aligned[: symbol_count * samples_per_symbol].reshape(symbol_count, samples_per_symbol).mean(axis=1)
    constellation = go.Figure(go.Scattergl(
        x=symbols.real,
        y=symbols.imag,
        mode="markers",
        marker={"color": COLORS[0], "size": 6, "opacity": 0.55},
        showlegend=False,
    ))
    default_constellation_limit = _comfortable_limit(symbols, 0.8)
    constellation_limit = float(metadata.get("examples:constellation_limit", default_constellation_limit))
    constellation.update_xaxes(
        title_text="In-phase",
        range=[-constellation_limit, constellation_limit],
        autorange=False,
        scaleanchor="y",
        scaleratio=1,
    )
    constellation.update_yaxes(
        title_text="Quadrature",
        range=[-constellation_limit, constellation_limit],
        autorange=False,
    )

    eye_length = 2 * samples_per_symbol
    eye_count = min(160, max(0, aligned.size // samples_per_symbol - 1))
    eye_segments = np.asarray([
        aligned[index * samples_per_symbol : index * samples_per_symbol + eye_length]
        for index in range(eye_count)
    ])
    eye_time = np.arange(eye_length) / samples_per_symbol
    eye_x = np.concatenate([np.append(eye_time, np.nan) for _ in range(eye_count)])
    eye = go.Figure()
    if eye_count:
        eye.add_trace(go.Scattergl(
            x=eye_x,
            y=np.concatenate([np.append(segment.real, np.nan) for segment in eye_segments]),
            name="I",
            mode="lines",
            line={"color": COLORS[0], "width": 1},
        ))
        eye.add_trace(go.Scattergl(
            x=eye_x,
            y=np.concatenate([np.append(segment.imag, np.nan) for segment in eye_segments]),
            name="Q",
            mode="lines",
            line={"color": COLORS[1], "width": 1},
        ))
    eye_limit = float(metadata.get("examples:eye_limit", _comfortable_limit(aligned, constellation_limit)))
    eye.update_xaxes(title_text="Symbol periods", range=[0, 2], autorange=False)
    eye.update_yaxes(title_text="Amplitude", range=[-eye_limit, eye_limit], autorange=False)

    ui.stat("Modulation", modulation)
    if symbol_rate is None:
        ui.stat("Coordinate basis", "Normalized samples")
        ui.stat("Samples per symbol", samples_per_symbol)
    else:
        ui.stat("Symbol rate", f"{symbol_rate / 1e3:g} ksym/s")
    with ui.tab("Constellation"):
        ui.plot(style_figure(constellation, ui.theme, f"{modulation} constellation"), key="constellation")
    with ui.tab("Eye diagram"):
        ui.plot(style_figure(eye, ui.theme, f"{modulation} eye diagram"), key="eye")


def _describe_recording(metadata_path: Path) -> DataResource:
    metadata = load_metadata(metadata_path)
    global_metadata = metadata["global"]
    modulation = _modulation_label(global_metadata, metadata_path.stem)
    raw_sample_rate = global_metadata.get("core:sample_rate")
    return DataResource(
        identifier=metadata_path.name.removesuffix(".sigmf-meta"),
        title=modulation,
        source=metadata_path,
        subtitle=(
            f"{float(raw_sample_rate):g} samples/s"
            if raw_sample_rate is not None
            else "Sample-normalized (rate unavailable)"
        ),
        timestamp=datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=timezone.utc),
        tags=("sigmf", str(global_metadata["core:datatype"]), modulation.lower()),
        summary=sigmf_discovery_summary(metadata),
    )


def create_workspace(config=None):
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data/comms"))
    return AnalysisWorkspace(
        identifier="digital-comms",
        name="Digital Communications",
        description="Windowed mode: compare file-backed QPSK and 16-QAM recordings with constellation and eye-diagram views.",
        source=DirectorySource(
            root,
            pattern="*.sigmf-meta",
            loader=lambda path: load_recording(path, sample_rate_fallback=1.0),
            describe=_describe_recording,
        ),
        delivery=WindowedCommsDelivery(),
        annotator=SigMFAnnotator(),
        exporter=SigMFExporter(),
        analyze=analyze,
        category="digital communications",
        tags=("windowed", "qpsk", "16-qam", "constellation", "eye diagram"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )


def _modulation_label(metadata: dict[str, object], fallback: str) -> str:
    explicit = metadata.get("examples:modulation")
    if explicit:
        return str(explicit)
    description = str(metadata.get("core:description") or fallback)
    upper = description.upper()
    if "16-QAM" in upper or "16QAM" in upper:
        return "16-QAM"
    if "QPSK" in upper:
        return "QPSK"
    return description


def _comfortable_limit(values: np.ndarray, fallback: float) -> float:
    values = np.asarray(values)
    if not values.size:
        return fallback
    extent = float(np.quantile(np.maximum(np.abs(values.real), np.abs(values.imag)), 0.995))
    return max(1e-6, extent * 1.15)
