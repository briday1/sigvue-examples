"""Profile factories for every workspace in this repository."""

from pathlib import Path

from .comms import create_workspace as create_comms_workspace
from .events import create_workspace as create_events_workspace
from .lfm_live import create_workspace as _lfm_live
from .lfm_static import create_workspace as _lfm_static
from .tones import create_workspace as create_tones_workspace


def create_lfm_live_workspace(config=None):
    values = config or {}
    return _lfm_live(
        Path(values.get("data_root", Path.cwd() / "data/lfm-collection")),
        identifier=str(values.get("id", "lfm-live")),
        name=str(values.get("name", "LFM Live")),
    )


def create_lfm_static_workspace(config=None):
    values = config or {}
    return _lfm_static(
        Path(values.get("data_root", Path.cwd() / "data/lfm-collection")),
        identifier=str(values.get("id", "lfm-static")),
        name=str(values.get("name", "LFM Static")),
    )

__all__ = [
    "create_comms_workspace",
    "create_events_workspace",
    "create_tones_workspace",
    "create_lfm_live_workspace",
    "create_lfm_static_workspace",
]
