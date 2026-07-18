"""Static whole-file entry point for the shared LFM analysis pipeline."""

from pathlib import Path

from .lfm_pipeline import WholeFileDelivery, create_lfm_workspace


def create_workspace(
    path: Path | None = None,
    *,
    identifier: str = "lfm-static",
    name: str = "LFM Static",
):
    return create_lfm_workspace(
        path,
        identifier=identifier,
        name=name,
        delivery=WholeFileDelivery(),
        description="Static mode: run the calibrated four-channel LFM analysis over the complete OTA files with no playback or buffering controls.",
        tags=("static", "whole file", "four-channel", "calibrated", "LFM"),
    )
