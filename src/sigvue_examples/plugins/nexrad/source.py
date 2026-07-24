"""Reusable Sigvue discovery for directories of NEXRAD Level III files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from sigvue.plugin import DataResource, DirectorySource, Source

from .models import (
    NexradLevel3Header,
    NexradLevel3Radial,
    NexradLevel3Sequence,
)
from .reader import read_level3_header, read_level3_radial


def describe_level3(path: Path) -> DataResource:
    header = read_level3_header(path)
    timestamp = header.scan_time
    return DataResource(
        identifier=path.name,
        title=(
            f"{header.radar_id} {header.product_id} · {timestamp:%Y-%m-%d %H:%M:%S} UTC"
        ),
        source=path,
        subtitle=(
            "NOAA NEXRAD Level III base reflectivity · "
            f"{header.elevation_deg:g}° elevation"
        ),
        timestamp=timestamp,
        tags=("NOAA", "NEXRAD", "Level III", header.product_id),
        summary={
            "date": timestamp.isoformat(),
            "sample_rate": None,
            "rf_frequency": None,
        },
    )


def level3_directory_source(root: str | Path) -> DirectorySource[NexradLevel3Radial]:
    """Discover native NODD names as well as conventional ``.nids`` files."""
    return DirectorySource(
        root,
        pattern=("*_N?B_*", "*.nids", "*.nids.gz"),
        loader=read_level3_radial,
        describe=describe_level3,
    )


class NexradLevel3SequenceSource(Source[NexradLevel3Sequence]):
    """Discover chronological scan collections grouped by directory and product."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def _paths(self) -> tuple[Path, ...]:
        patterns = ("*_N?B_*", "*.nids", "*.nids.gz")
        return tuple(
            sorted(
                {
                    path.resolve()
                    for pattern in patterns
                    for path in self.root.rglob(pattern)
                    if path.is_file()
                }
            )
        )

    def discover(self) -> tuple[DataResource, ...]:
        groups: dict[
            tuple[Path, str, str],
            list[NexradLevel3Header],
        ] = defaultdict(list)
        for path in self._paths():
            header = read_level3_header(path)
            groups[(path.parent, header.radar_id, header.product_id)].append(header)

        resources = []
        for (directory, radar_id, product_id), headers in sorted(
            groups.items(),
            key=lambda item: (
                str(item[0][0]),
                item[0][1],
                item[0][2],
            ),
        ):
            sequence = NexradLevel3Sequence(
                tuple(sorted(headers, key=lambda header: header.scan_time))
            )
            relative = directory.relative_to(self.root)
            navigation_path = () if relative == Path(".") else relative.parts
            start = sequence.headers[0].scan_time
            stop = sequence.headers[-1].scan_time
            resources.append(
                DataResource(
                    identifier=(
                        f"{relative.as_posix()}::{radar_id}-{product_id}"
                        if navigation_path
                        else f"{radar_id}-{product_id}"
                    ),
                    title=f"{radar_id} {product_id} reflectivity sequence",
                    source=sequence,
                    subtitle=(
                        f"{sequence.scan_count} scans · "
                        f"{start:%Y-%m-%d %H:%M:%S} to "
                        f"{stop:%H:%M:%S} UTC"
                    ),
                    timestamp=start,
                    tags=(
                        "NOAA",
                        "NEXRAD",
                        "Level III",
                        product_id,
                        "time sequence",
                    ),
                    summary={
                        "date": start.isoformat(),
                        "sample_rate": None,
                        "rf_frequency": None,
                    },
                    navigation_path=navigation_path,
                )
            )
        return tuple(resources)

    def open(self, resource: DataResource) -> NexradLevel3Sequence:
        if not isinstance(resource.source, NexradLevel3Sequence):
            raise TypeError("NEXRAD sequence resources must contain a sequence")
        return resource.source


def level3_sequence_source(root: str | Path) -> NexradLevel3SequenceSource:
    """Create a source that exposes each radar/product group as one timeline."""
    return NexradLevel3SequenceSource(root)


__all__ = [
    "NexradLevel3SequenceSource",
    "describe_level3",
    "level3_directory_source",
    "level3_sequence_source",
]
