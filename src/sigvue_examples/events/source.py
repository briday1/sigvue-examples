"""Stored-result discovery and JSON loading."""

from datetime import datetime, timezone
import json
from pathlib import Path

from sigvue.plugin import DataResource, DirectorySource

from .models import AcousticEventCollection, StoredEventResults


def load_collection(path: Path) -> AcousticEventCollection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = tuple(
        StoredEventResults(
            identifier=str(event["id"]), label=str(event["label"]),
            start_seconds=float(event["start_seconds"]),
            duration_seconds=float(event["duration_seconds"]),
            confidence=float(event["confidence"]),
            waveform_time=tuple(float(value) for value in event["waveform_time"]),
            waveform=tuple(float(value) for value in event["waveform"]),
            spectrum_frequency=tuple(float(value) for value in event["spectrum_frequency"]),
            spectrum_db=tuple(float(value) for value in event["spectrum_db"]),
        )
        for event in payload["events"]
    )
    return AcousticEventCollection(path, float(payload["duration_seconds"]), events)


def describe_collection(path: Path) -> DataResource:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DataResource(
        identifier=path.stem, title=str(payload.get("title", path.stem)), source=path,
        subtitle=f"{len(payload['events'])} precomputed events · {float(payload['duration_seconds']):g} s",
        timestamp=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
        tags=("json", "precomputed", "acoustic events"),
        summary={"date": None, "sample_rate": None, "rf_frequency": None},
    )


def event_source(root: Path) -> DirectorySource:
    return DirectorySource(
        root,
        pattern="acoustic-events.json",
        loader=load_collection,
        describe=describe_collection,
    )


__all__ = ["AcousticEventCollection", "event_source"]
