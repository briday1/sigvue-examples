"""Workspace assembly for stored acoustic-event results."""

from pathlib import Path

from sigvue.plugin import Workspace

from ..io.sigmf.capabilities import SIGNAL_DISCOVERY_COLUMNS
from .analysis import EventAnalysis
from .delivery import StoredEventDelivery
from .presentation import EventPresentation
from .source import event_source


def create_workspace(config=None) -> Workspace:
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data"))
    return Workspace(
        identifier="acoustic-events-segmented",
        name="Acoustic Event Review",
        description="Segmented mode: navigate irregular precomputed acoustic events and display stored results without reprocessing raw data.",
        source=event_source(root),
        delivery=StoredEventDelivery(),
        analysis=EventAnalysis(),
        presentation=EventPresentation(),
        category="acoustic monitoring",
        tags=("segmented", "irregular events", "precomputed", "display-only"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )
