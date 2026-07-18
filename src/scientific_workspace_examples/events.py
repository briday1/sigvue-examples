"""Display-only review of irregular, precomputed acoustic events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import plotly.graph_objects as go

from workspace_browser.plugin import AnalysisContext, AnalysisWorkspace, DataDelivery, DataResource, DirectorySource, Segment

from .style import COLORS, style_figure


@dataclass(frozen=True)
class StoredEventResults:
    identifier: str
    label: str
    start_seconds: float
    duration_seconds: float
    confidence: float
    waveform_time: tuple[float, ...]
    waveform: tuple[float, ...]
    spectrum_frequency: tuple[float, ...]
    spectrum_db: tuple[float, ...]


@dataclass(frozen=True)
class AcousticEventCollection:
    path: Path
    duration_seconds: float
    events: tuple[StoredEventResults, ...]


class StoredEventDelivery(DataDelivery[AcousticEventCollection, StoredEventResults]):
    """Select one stored result; no signal processing occurs in the workspace."""

    def prepare(self, collection: AcousticEventCollection, ui: AnalysisContext) -> StoredEventResults:
        selected = ui.segmented(
            duration=collection.duration_seconds,
            segments=tuple(
                Segment(event.identifier, event.start_seconds, event.duration_seconds, event.label)
                for event in collection.events
            ),
        )
        return next(event for event in collection.events if event.identifier == selected.identifier)


def analyze(event: StoredEventResults, ui: AnalysisContext) -> None:
    ui.stat("Stored event", event.label)
    ui.stat("Confidence", f"{event.confidence:.1%}")
    ui.stat("Start", f"{event.start_seconds:.3f} s")
    ui.stat("Duration", f"{event.duration_seconds:.3f} s")

    waveform = go.Figure(go.Scatter(
        x=event.waveform_time,
        y=event.waveform,
        mode="lines",
        line={"color": COLORS[0], "width": 1.5},
        showlegend=False,
    ))
    waveform.update_xaxes(title_text="Time within event (s)", range=[0, event.duration_seconds])
    waveform.update_yaxes(title_text="Stored normalized amplitude", range=[-1.1, 1.1])

    spectrum = go.Figure(go.Scatter(
        x=event.spectrum_frequency,
        y=event.spectrum_db,
        mode="lines",
        line={"color": COLORS[1], "width": 1.5},
        showlegend=False,
    ))
    spectrum.update_xaxes(title_text="Frequency (Hz)")
    spectrum.update_yaxes(title_text="Stored magnitude (dB)", range=[-80, 5])

    with ui.tab("Stored waveform"):
        ui.plot(style_figure(waveform, ui.theme, event.label), key="waveform")
    with ui.tab("Stored spectrum"):
        ui.plot(style_figure(spectrum, ui.theme, f"{event.label} spectrum"), key="spectrum")


def create_workspace(config=None):
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data"))
    return AnalysisWorkspace(
        identifier=str(values.get("id", "acoustic-events-segmented")),
        name=str(values.get("name", "Acoustic Event Review")),
        description="Segmented mode: navigate irregular precomputed acoustic events and display stored results without reprocessing raw data.",
        source=DirectorySource(
            root,
            pattern="acoustic-events.json",
            loader=load_collection,
            describe=describe_collection,
        ),
        delivery=StoredEventDelivery(),
        analyze=analyze,
        category="acoustic monitoring",
        tags=("segmented", "irregular events", "precomputed", "display-only"),
    )


def load_collection(path: Path) -> AcousticEventCollection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = tuple(
        StoredEventResults(
            identifier=str(event["id"]),
            label=str(event["label"]),
            start_seconds=float(event["start_seconds"]),
            duration_seconds=float(event["duration_seconds"]),
            confidence=float(event["confidence"]),
            waveform_time=tuple(float(value) for value in event["waveform_time"]),
            waveform=tuple(float(value) for value in event["waveform"]),
            spectrum_frequency=tuple(float(value) for value in event["spectrum_frequency"]),
            spectrum_db=tuple(float(value) for value in event["spectrum_db"]),
        )
        for event in payload["events"]
    )
    return AcousticEventCollection(path, float(payload["duration_seconds"]), events)


def describe_collection(path: Path) -> DataResource:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DataResource(
        identifier=path.stem,
        title=str(payload.get("title", path.stem)),
        source=path,
        subtitle=f"{len(payload['events'])} precomputed events · {float(payload['duration_seconds']):g} s",
        timestamp=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
        tags=("json", "precomputed", "acoustic events"),
    )
