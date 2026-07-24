"""Drop-in discovery for directories of local WFDB records."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from sigvue.plugin import DataResource, DirectorySource, DiscoveryColumn

from .recording import WFDBRecording, load_wfdb_record, parse_wfdb_header
from .annotations import read_mit_annotations


WFDB_DISCOVERY_COLUMNS = (
    DiscoveryColumn("sample_rate", "Sampling rate", "si", unit="sample/s"),
    DiscoveryColumn("duration_seconds", "Duration (s)", "number"),
    DiscoveryColumn("leads", "Leads", "text"),
    DiscoveryColumn("annotation_count", "Annotations", "number"),
)


def describe_wfdb_record(
    path: Path,
    *,
    tags: Iterable[str] = ("WFDB",),
) -> DataResource:
    """Build a catalog item from header facts without reading signal data."""
    header = parse_wfdb_header(path)
    annotation_path = path.with_suffix(".atr")
    annotation_count = (
        len(read_mit_annotations(annotation_path)) if annotation_path.is_file() else 0
    )
    duration_minutes = header.duration_seconds / 60
    return DataResource(
        identifier=header.record_name,
        title=f"WFDB record {header.record_name}",
        source=path,
        subtitle=(
            f"{header.channel_count} leads · {header.sample_rate:g} Hz · "
            f"{duration_minutes:g} min · {annotation_count:,} annotations"
        ),
        timestamp=datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        ),
        tags=tuple(tags),
        summary={
            "sample_rate": header.sample_rate,
            "duration_seconds": header.duration_seconds,
            "leads": ", ".join(channel.name for channel in header.channels),
            "annotation_count": annotation_count,
        },
    )


def wfdb_source(
    directory: str | Path,
    *,
    pattern: str | tuple[str, ...] = "*.hea",
    recursive: bool = True,
    tags: Iterable[str] = ("WFDB",),
) -> DirectorySource[WFDBRecording]:
    """Create a complete local WFDB source with exact record loading."""
    root = Path(directory).expanduser().resolve()

    def descriptor(path: Path) -> DataResource:
        resource = describe_wfdb_record(path, tags=tags)
        relative = path.resolve().relative_to(root)
        parent = relative.parent
        identifier = relative.as_posix().removesuffix(".hea")
        return replace(
            resource,
            identifier=identifier.replace("/", "::"),
            navigation_path=() if parent == Path(".") else parent.parts,
        )

    return DirectorySource(
        root,
        pattern=pattern,
        loader=load_wfdb_record,
        describe=descriptor,
        recursive=recursive,
    )


__all__ = [
    "WFDB_DISCOVERY_COLUMNS",
    "describe_wfdb_record",
    "wfdb_source",
]
