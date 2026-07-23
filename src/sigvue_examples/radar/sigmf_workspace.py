"""LFM workspace for standard SigMF multi-stream captures."""

from pathlib import Path

from sigvue.plugin import Workspace

from ..io.sigmf.capabilities import SIGNAL_DISCOVERY_COLUMNS
from .analysis import LfmAnalysis
from .capabilities import LfmAnnotator, LfmExporter
from .delivery import BufferedDelivery
from .presentation import LfmPresentation
from .sigmf_source import sigmf_collection_source


def create_workspace(config=None) -> Workspace:
    values = config or {}
    root = Path(values.get("data_root", Path.cwd() / "data/lfm-sigmf"))
    source = sigmf_collection_source(
        root,
        calibration_dbm=float(values.get("calibration_dbm", -20.0)),
        ota_prf_hz=float(values.get("ota_prf_hz", 1_000.0)),
        ota_pulse_width_seconds=float(values.get("ota_pulse_width_seconds", 50e-6)),
    )
    return Workspace(
        identifier="lfm-sigmf",
        name="LFM SigMF View",
        delivery=BufferedDelivery(),
        description="Inspect standard multi-stream SigMF captures with the shared buffered LFM analysis pipeline.",
        source=source,
        annotator=LfmAnnotator(),
        exporter=LfmExporter(),
        analysis=LfmAnalysis(),
        presentation=LfmPresentation(),
        lazy_views=True,
        category="signal analysis",
        tags=("live", "multi-channel", "SigMF", "LFM", "capture"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )


__all__ = ["create_workspace"]
