"""Drop-in helpers for NOAA NEXRAD Level III base-reflectivity files."""

from .delivery import SegmentedNexradDelivery
from .models import (
    BELOW_THRESHOLD_CODE,
    FIRST_MEASURED_CODE,
    NexradLevel3Header,
    NexradLevel3Radial,
    NexradLevel3Sequence,
    NexradSequenceSelection,
    RANGE_FOLDED_CODE,
)
from .reader import (
    NexradFormatError,
    PACKET_CODE_DIGITAL_RADIAL,
    PRODUCT_CODE_SUPER_RESOLUTION_REFLECTIVITY,
    read_level3_header,
    read_level3_radial,
)
from .source import (
    NexradLevel3SequenceSource,
    describe_level3,
    level3_directory_source,
    level3_sequence_source,
)

__all__ = [
    "BELOW_THRESHOLD_CODE",
    "FIRST_MEASURED_CODE",
    "NexradFormatError",
    "NexradLevel3Header",
    "NexradLevel3Radial",
    "NexradLevel3Sequence",
    "NexradLevel3SequenceSource",
    "NexradSequenceSelection",
    "PACKET_CODE_DIGITAL_RADIAL",
    "PRODUCT_CODE_SUPER_RESOLUTION_REFLECTIVITY",
    "RANGE_FOLDED_CODE",
    "SegmentedNexradDelivery",
    "describe_level3",
    "level3_directory_source",
    "level3_sequence_source",
    "read_level3_header",
    "read_level3_radial",
]
