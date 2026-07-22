"""Reusable channel-layout utilities for radar plots."""

from dataclasses import dataclass
from math import ceil, sqrt


@dataclass(frozen=True)
class ChannelGrid:
    """A compact, near-square row-major grid for a channel collection."""

    channel_count: int
    rows: int
    columns: int

    def position(self, channel_index: int) -> tuple[int, int]:
        if not 0 <= channel_index < self.channel_count:
            raise IndexError(f"Channel index {channel_index} is outside this grid")
        row, column = divmod(channel_index, self.columns)
        return row + 1, column + 1


def channel_grid(channel_count: int) -> ChannelGrid:
    """Return a near-square grid: four channels are 2×2, sixteen are 4×4."""
    if channel_count < 1:
        raise ValueError("A channel grid requires at least one channel")
    columns = ceil(sqrt(channel_count))
    rows = ceil(channel_count / columns)
    return ChannelGrid(channel_count, rows, columns)
