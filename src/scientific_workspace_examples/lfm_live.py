"""Live buffered entry point for the shared LFM analysis pipeline."""

from pathlib import Path

from .lfm_pipeline import BufferedDelivery, create_lfm_workspace


def create_workspace(
    path: Path | None = None,
    *,
    identifier: str = "lfm-live",
    name: str = "LFM Live",
):
    return create_lfm_workspace(
        path,
        identifier=identifier,
        name=name,
        delivery=BufferedDelivery(),
        description="Follow a growing four-channel LFM recording live, or seek through history using buffered reads and shared calibration.",
        tags=("live", "four-channel", "calibrated", "LFM", "waterfall"),
    )
