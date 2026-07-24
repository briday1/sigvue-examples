"""Framework assembly for the annotated ECG workspace."""

from sigvue.helpers import WorkspaceConfig
from sigvue.plugin import Workspace

from ..plugins import CallableAnalysis, CallablePresentation
from ..plugins.wfdb import (
    WFDB_DISCOVERY_COLUMNS,
    WindowedWFDBDelivery,
    wfdb_source,
)
from .analysis import process
from .presentation import present


def create_workspace(config=None) -> Workspace:
    values = WorkspaceConfig(config)
    root = values.path("data_root", "data/ecg/mit-bih")
    return Workspace(
        identifier="mit-bih-ecg",
        name="Annotated ECG",
        description=(
            "Exact two-lead MIT-BIH waveforms with cardiologist reference "
            "annotations, RR intervals, and beat morphology."
        ),
        source=wfdb_source(
            root,
            pattern=values.string("filename", "*.hea"),
            tags=("WFDB", "MIT-BIH", "ECG", "reference annotations"),
        ),
        delivery=WindowedWFDBDelivery(
            default_window=10.0,
            minimum_window=2.0,
            step=1.0,
            overview_bins=360,
            overview_label="MLII peak-to-peak amplitude (mV)",
            cache_key="mit-bih-ecg-overview",
        ),
        analysis=CallableAnalysis(process),
        presentation=CallablePresentation(present),
        lazy_views=True,
        category="physiology",
        tags=("windowed", "ECG", "WFDB", "real data", "annotations"),
        discovery_columns=WFDB_DISCOVERY_COLUMNS,
    )


__all__ = ["create_workspace"]
