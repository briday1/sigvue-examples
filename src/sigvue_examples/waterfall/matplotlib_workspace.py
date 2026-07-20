"""Workspace assembly for the Matplotlib waterfall example."""

from pathlib import Path

from sigvue.plugin import Workspace

from ..io.sigmf.capabilities import SIGNAL_DISCOVERY_COLUMNS
from .analysis import WaterfallAnalysis
from .capabilities import waterfall_capabilities
from .delivery import WindowedWaterfallDelivery
from .matplotlib_plots import MatplotlibWaterfallPresentation
from .source import waterfall_source


def create_workspace(config=None) -> Workspace:
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data"))
    filename = str(values.get("filename", "*.sigmf-meta"))
    source_type = str(values.get("source_type", "recording"))
    source, is_collection = waterfall_source(root, filename, source_type)
    _, exporter = waterfall_capabilities(is_collection)
    return Workspace(
        identifier="matplotlib-waterfall",
        name="Matplotlib Waterfall",
        description="Windowed SigMF spectrum and waterfall rendered as a static Matplotlib PNG.",
        source=source,
        delivery=WindowedWaterfallDelivery(),
        exporter=exporter,
        analysis=WaterfallAnalysis(),
        presentation=MatplotlibWaterfallPresentation(),
        category="spectrum monitoring",
        tags=("windowed", "sigmf", "waterfall", "matplotlib", "static"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )
