"""Assembly of the NOAA NEXRAD weather-radar workspace."""

from sigvue.helpers import WorkspaceConfig
from sigvue.plugin import Workspace

from ..plugins import (
    CallableAnalysis,
    CallablePresentation,
    SIGNAL_DISCOVERY_COLUMNS,
)
from ..plugins.nexrad import SegmentedNexradDelivery, level3_sequence_source
from .analysis import process
from .presentation import present


def create_workspace(config=None) -> Workspace:
    values = WorkspaceConfig(config)
    root = values.path("data_root", "data/weather-radar")
    return Workspace(
        identifier="weather-radar",
        name="NOAA Weather Radar",
        description=(
            "Explore exact native gates from dense two-hour NOAA NEXRAD "
            "Level III base-reflectivity sequences, with segmented time "
            "navigation and display-only PPI resampling."
        ),
        source=level3_sequence_source(root),
        delivery=SegmentedNexradDelivery(),
        analysis=CallableAnalysis(process),
        presentation=CallablePresentation(present),
        lazy_views=True,
        category="weather radar",
        tags=("NOAA", "NEXRAD", "Level III", "base reflectivity"),
        discovery_columns=SIGNAL_DISCOVERY_COLUMNS,
    )


__all__ = ["create_workspace"]
