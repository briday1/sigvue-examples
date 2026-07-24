"""Segmented delivery for chronological NEXRAD Level III sequences."""

from __future__ import annotations

from sigvue.plugin import Delivery, DeliveryContext, Segment

from .models import NexradLevel3Sequence, NexradSequenceSelection
from .reader import read_level3_radial


class SegmentedNexradDelivery(Delivery[NexradLevel3Sequence, NexradSequenceSelection]):
    """Expose each discrete radar scan as one previous/next segment."""

    def __init__(self, *, cache_key: str = "nexrad-level3-scan") -> None:
        if not cache_key:
            raise ValueError("cache_key cannot be empty")
        self.cache_key = cache_key

    def prepare(
        self,
        sequence: NexradLevel3Sequence,
        ui: DeliveryContext,
    ) -> NexradSequenceSelection:
        elapsed = sequence.elapsed_seconds
        durations = tuple(
            elapsed[index + 1] - elapsed[index]
            if index + 1 < sequence.scan_count
            else sequence.nominal_interval_seconds
            for index in range(sequence.scan_count)
        )
        segments = tuple(
            Segment(
                identifier=header.source_path.name,
                start_seconds=elapsed[index],
                duration_seconds=durations[index],
                label=f"{header.scan_time:%H:%M:%S} UTC",
            )
            for index, header in enumerate(sequence.headers)
        )
        selected = ui.segmented(
            duration=elapsed[-1] + durations[-1],
            segments=segments,
            time_unit="min",
        )
        scan_index = next(
            index
            for index, segment in enumerate(segments)
            if segment.identifier == selected.identifier
        )
        header = sequence.headers[scan_index]
        scan = ui.once(
            f"{self.cache_key}:{header.source_path}",
            lambda: read_level3_radial(header.source_path),
        )
        return NexradSequenceSelection(
            sequence=sequence,
            scan_index=scan_index,
            scan=scan,
        )


__all__ = ["SegmentedNexradDelivery"]
