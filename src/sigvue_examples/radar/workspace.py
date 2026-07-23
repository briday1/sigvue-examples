"""Workspace assembly for the live calibrated LFM pipeline."""

from pathlib import Path

from sigvue.plugin import Workspace

from ..io.sigmf.capabilities import SIGNAL_DISCOVERY_COLUMNS
from .analysis import LfmAnalysis
from .capabilities import LfmAnnotator, LfmExporter
from .delivery import BufferedDelivery
from .presentation import LfmPresentation
from .source import collection_source


def create_workspace(config=None) -> Workspace:
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data/lfm-live"))
    return Workspace(
        identifier="lfm-live",
        name="LFM Live View",
        delivery=BufferedDelivery(),
        description="Choose a four- or sixteen-channel 10 MHz collection or a 2 MHz multi-target collection, then follow it live or seek through history using the same buffered calibration analysis.",
        source=collection_source(root),
        annotator=LfmAnnotator(),
        exporter=LfmExporter(),
        analysis=LfmAnalysis(),
        presentation=LfmPresentation(),
        lazy_views=True,
        category="signal analysis",
        tags=("live", "multi-channel", "four-channel", "sixteen-channel", "calibrated", "LFM", "10-mhz", "2-mhz", "multi-target", "waterfall"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )
