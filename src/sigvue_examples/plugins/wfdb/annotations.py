"""Exact readers for compact MIT-format WFDB annotation files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WFDBAnnotationLabel:
    """The standard display information for one WFDB annotation code."""

    symbol: str
    short_name: str
    description: str
    is_beat: bool = False


@dataclass(frozen=True)
class WFDBAnnotation:
    """One annotation located at an exact native sample index."""

    sample: int
    code: int
    symbol: str
    short_name: str
    description: str
    subtype: int = 0
    channel: int = 0
    number: int = 0
    auxiliary_note: str = ""
    is_beat: bool = False

    def time_seconds(self, sample_rate: float) -> float:
        """Return elapsed time without discarding the authoritative sample."""
        return self.sample / sample_rate


# These values are the standard codes from WFDB's ecgcodes.h. The beat flag
# follows the physiologic QRS-like labels used for RR interval calculations;
# rhythm, signal-quality, and waveform-boundary annotations are deliberately
# excluded.
ANNOTATION_LABELS: dict[int, WFDBAnnotationLabel] = {
    1: WFDBAnnotationLabel("N", "NORMAL", "Normal beat", True),
    2: WFDBAnnotationLabel("L", "LBBB", "Left bundle branch block beat", True),
    3: WFDBAnnotationLabel("R", "RBBB", "Right bundle branch block beat", True),
    4: WFDBAnnotationLabel(
        "a",
        "ABERR",
        "Aberrated atrial premature beat",
        True,
    ),
    5: WFDBAnnotationLabel(
        "V",
        "PVC",
        "Premature ventricular contraction",
        True,
    ),
    6: WFDBAnnotationLabel(
        "F",
        "FUSION",
        "Fusion of ventricular and normal beat",
        True,
    ),
    7: WFDBAnnotationLabel(
        "J",
        "NPC",
        "Nodal (junctional) premature beat",
        True,
    ),
    8: WFDBAnnotationLabel(
        "A",
        "APC",
        "Atrial premature contraction",
        True,
    ),
    9: WFDBAnnotationLabel(
        "S",
        "SVPB",
        "Premature or ectopic supraventricular beat",
        True,
    ),
    10: WFDBAnnotationLabel(
        "E",
        "VESC",
        "Ventricular escape beat",
        True,
    ),
    11: WFDBAnnotationLabel(
        "j",
        "NESC",
        "Nodal (junctional) escape beat",
        True,
    ),
    12: WFDBAnnotationLabel("/", "PACE", "Paced beat", True),
    13: WFDBAnnotationLabel("Q", "UNKNOWN", "Unclassifiable beat", True),
    14: WFDBAnnotationLabel("~", "NOISE", "Signal quality change"),
    16: WFDBAnnotationLabel("|", "ARFCT", "Isolated QRS-like artifact"),
    18: WFDBAnnotationLabel("s", "STCH", "ST change"),
    19: WFDBAnnotationLabel("T", "TCH", "T-wave change"),
    20: WFDBAnnotationLabel("*", "SYSTOLE", "Systole"),
    21: WFDBAnnotationLabel("D", "DIASTOLE", "Diastole"),
    22: WFDBAnnotationLabel('"', "NOTE", "Comment annotation"),
    23: WFDBAnnotationLabel("=", "MEASURE", "Measurement annotation"),
    24: WFDBAnnotationLabel("p", "PWAVE", "P-wave peak"),
    25: WFDBAnnotationLabel(
        "B",
        "BBB",
        "Left or right bundle branch block beat",
        True,
    ),
    26: WFDBAnnotationLabel("^", "PACESP", "Non-conducted pacer spike"),
    27: WFDBAnnotationLabel("t", "TWAVE", "T-wave peak"),
    28: WFDBAnnotationLabel("+", "RHYTHM", "Rhythm change"),
    29: WFDBAnnotationLabel("u", "UWAVE", "U-wave peak"),
    30: WFDBAnnotationLabel("?", "LEARN", "Learning"),
    31: WFDBAnnotationLabel("!", "FLWAV", "Ventricular flutter wave"),
    32: WFDBAnnotationLabel(
        "[",
        "VFON",
        "Start of ventricular flutter/fibrillation",
    ),
    33: WFDBAnnotationLabel(
        "]",
        "VFOFF",
        "End of ventricular flutter/fibrillation",
    ),
    34: WFDBAnnotationLabel("e", "AESC", "Atrial escape beat", True),
    35: WFDBAnnotationLabel(
        "n",
        "SVESC",
        "Supraventricular escape beat",
        True,
    ),
    36: WFDBAnnotationLabel(
        "@",
        "LINK",
        "Link to external data (auxiliary note contains URL)",
    ),
    37: WFDBAnnotationLabel(
        "x",
        "NAPC",
        "Non-conducted P-wave (blocked APB)",
    ),
    38: WFDBAnnotationLabel(
        "f",
        "PFUS",
        "Fusion of paced and normal beat",
        True,
    ),
    39: WFDBAnnotationLabel("(", "WFON", "Waveform onset"),
    40: WFDBAnnotationLabel(")", "WFOFF", "Waveform end"),
    41: WFDBAnnotationLabel(
        "r",
        "RONT",
        "R-on-T premature ventricular contraction",
        True,
    ),
}

_SKIP = 59
_NUM = 60
_SUB = 61
_CHAN = 62
_AUX = 63


def _signed_byte(value: int) -> int:
    return value - 256 if value >= 128 else value


def _require_pairs(payload: bytes, path: Path) -> memoryview:
    if len(payload) % 2:
        raise ValueError(f"{path.name} ends with a partial annotation word")
    return memoryview(payload)


def _pair(payload: memoryview, index: int, path: Path) -> tuple[int, int]:
    offset = 2 * index
    if offset + 2 > len(payload):
        raise ValueError(f"{path.name} ended during annotation decoding")
    return int(payload[offset]), int(payload[offset + 1])


def _skip_interval(
    payload: memoryview,
    pair_index: int,
    path: Path,
) -> int:
    """Decode WFDB's PDP-11-order signed 32-bit SKIP interval."""
    high_lo, high_hi = _pair(payload, pair_index + 1, path)
    low_lo, low_hi = _pair(payload, pair_index + 2, path)
    value = (high_lo << 16) | (high_hi << 24) | low_lo | (low_hi << 8)
    return value - (1 << 32) if value >= (1 << 31) else value


def read_mit_annotations(path: str | Path) -> tuple[WFDBAnnotation, ...]:
    """Read an MIT-format annotation file without altering sample indices.

    The compact format stores ordinary time deltas in ten bits. ``SKIP``
    words extend that range with a signed 32-bit interval; ``NUM``, ``SUB``,
    ``CHAN``, and ``AUX`` words attach metadata to the preceding core
    annotation. This function implements each of those standard fields.
    """
    annotation_path = Path(path)
    payload = _require_pairs(annotation_path.read_bytes(), annotation_path)
    pair_count = len(payload) // 2
    pair_index = 0
    sample_total = 0
    current_channel = 0
    current_number = 0
    annotations: list[WFDBAnnotation] = []

    while pair_index < pair_count:
        sample_delta = 0
        first, second = _pair(payload, pair_index, annotation_path)
        code = second >> 2

        while code == _SKIP:
            if (first | (second & 0x03)) != 0:
                raise ValueError(f"{annotation_path.name} has a malformed SKIP word")
            sample_delta += _skip_interval(
                payload,
                pair_index,
                annotation_path,
            )
            pair_index += 3
            first, second = _pair(payload, pair_index, annotation_path)
            code = second >> 2

        interval = first | ((second & 0x03) << 8)
        if code == 0 and interval == 0:
            break
        if not 0 < code < _SKIP:
            raise ValueError(
                f"{annotation_path.name} has an unexpected core code {code}"
            )
        sample_delta += interval
        sample_total += sample_delta
        if sample_total < 0:
            raise ValueError(f"{annotation_path.name} moves before sample zero")
        pair_index += 1

        subtype = 0
        channel = current_channel
        number = current_number
        auxiliary_note = ""
        while pair_index < pair_count:
            extra_first, extra_second = _pair(
                payload,
                pair_index,
                annotation_path,
            )
            extra_code = extra_second >> 2
            if extra_code <= _SKIP:
                break
            if extra_code == _SUB:
                subtype = _signed_byte(extra_first)
                pair_index += 1
            elif extra_code == _CHAN:
                channel = extra_first
                current_channel = channel
                pair_index += 1
            elif extra_code == _NUM:
                number = _signed_byte(extra_first)
                current_number = number
                pair_index += 1
            elif extra_code == _AUX:
                length = extra_first
                byte_offset = 2 * (pair_index + 1)
                padded_length = length + (length & 1)
                if byte_offset + padded_length > len(payload):
                    raise ValueError(f"{annotation_path.name} ended in auxiliary text")
                auxiliary_note = (
                    bytes(payload[byte_offset : byte_offset + length])
                    .decode("latin-1")
                    .rstrip("\x00")
                )
                pair_index += 1 + padded_length // 2
            else:  # pragma: no cover - all six-bit values are enumerated
                raise AssertionError("unreachable")

        label = ANNOTATION_LABELS.get(
            code,
            WFDBAnnotationLabel(
                str(code),
                f"CODE_{code}",
                f"WFDB annotation code {code}",
            ),
        )
        annotations.append(
            WFDBAnnotation(
                sample=sample_total,
                code=code,
                symbol=label.symbol,
                short_name=label.short_name,
                description=label.description,
                subtype=subtype,
                channel=channel,
                number=number,
                auxiliary_note=auxiliary_note,
                is_beat=label.is_beat,
            )
        )

    return tuple(annotations)


__all__ = [
    "ANNOTATION_LABELS",
    "WFDBAnnotation",
    "WFDBAnnotationLabel",
    "read_mit_annotations",
]
