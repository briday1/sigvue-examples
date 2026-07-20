"""Workspace assembly for the reusable SigMF waterfall pipeline."""

from pathlib import Path

from sigvue.plugin import Workspace

from ..io.sigmf.capabilities import SIGNAL_DISCOVERY_COLUMNS
from .analysis import WaterfallAnalysis
from .batch import WaterfallBatch
from .capabilities import waterfall_capabilities
from .delivery import WindowedWaterfallDelivery
from .plots import WaterfallPresentation
from .source import waterfall_source


def create_workspace(config=None) -> Workspace:
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data"))
    filename = str(values.get("filename", "*.sigmf-meta"))
    source_type = str(values.get("source_type", "recording"))
    source, is_collection = waterfall_source(root, filename, source_type)
    annotator, exporter = waterfall_capabilities(is_collection)
    return Workspace(
        identifier="waterfall",
        name="Waterfall",
        description="Windowed mode: inspect a SigMF recording as an average spectrum and waterfall.",
        source=source,
        delivery=WindowedWaterfallDelivery(),
        annotator=annotator,
        exporter=exporter,
        batch=WaterfallBatch(),
        analysis=WaterfallAnalysis(),
        presentation=WaterfallPresentation(),
        category="spectrum monitoring",
        tags=("windowed", "sigmf", "spectrogram", "waterfall"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )
