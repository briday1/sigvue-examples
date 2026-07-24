"""Framework-independent WFDB header parsing and exact ranged sample I/O."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re

import numpy as np

from .annotations import WFDBAnnotation, read_mit_annotations


_FORMAT_PATTERN = re.compile(
    r"^(?P<format>\d+)(?:x(?P<samples_per_frame>\d+))?"
    r"(?::(?P<skew>-?\d+))?(?:\+(?P<byte_offset>\d+))?$"
)
_GAIN_PATTERN = re.compile(
    r"^(?P<gain>[-+]?\d+(?:\.\d+)?)?"
    r"(?:\((?P<baseline>[-+]?\d+)\))?"
    r"(?:/(?P<units>.*))?$"
)


@dataclass(frozen=True)
class WFDBChannel:
    """Signal metadata from one WFDB header signal line."""

    file_name: str
    format: int
    samples_per_frame: int
    skew: int
    byte_offset: int
    gain: float
    baseline: int
    units: str
    adc_resolution: int
    adc_zero: int
    initial_value: int
    checksum: int
    block_size: int
    name: str


@dataclass(frozen=True)
class WFDBHeader:
    """Validated metadata for a single-segment WFDB record."""

    path: Path
    record_name: str
    sample_rate: float
    sample_count: int
    channels: tuple[WFDBChannel, ...]
    comments: tuple[str, ...]

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate


@dataclass(frozen=True)
class WFDBRecording:
    """A WFDB recording with exact native samples and annotations."""

    header: WFDBHeader
    data_path: Path
    annotation_path: Path | None
    annotations: tuple[WFDBAnnotation, ...]

    @property
    def record_name(self) -> str:
        return self.header.record_name

    @property
    def sample_rate(self) -> float:
        return self.header.sample_rate

    @property
    def sample_count(self) -> int:
        return self.header.sample_count

    @property
    def channel_count(self) -> int:
        return self.header.channel_count

    @property
    def duration_seconds(self) -> float:
        return self.header.duration_seconds

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.header.channels)

    def read_digital(self, start: int, count: int) -> np.ndarray:
        """Read a channel-first native integer sample window.

        Format 212 is decoded from its interleaved 12-bit byte triplets
        without scaling or rounding. Format 16 is also supported because it
        is the other common, directly addressable WFDB representation.
        """
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(count, bool)
            or not isinstance(count, int)
        ):
            raise TypeError("WFDB ranged reads require integer start and count")
        if start < 0 or start > self.sample_count or count < 0:
            raise ValueError("WFDB ranged read is outside the recording")
        count = min(count, self.sample_count - start)
        if count == 0:
            return np.empty((self.channel_count, 0), dtype=np.int16)

        formats = {channel.format for channel in self.header.channels}
        if len(formats) != 1:
            raise ValueError("Mixed WFDB formats are not supported")
        data_format = formats.pop()
        if data_format == 212:
            return self._read_format_212(start, count)
        if data_format == 16:
            return self._read_format_16(start, count)
        raise ValueError(f"Unsupported WFDB signal format: {data_format}")

    def read_physical(self, start: int, count: int) -> np.ndarray:
        """Read calibrated values using ``(digital - baseline) / gain``."""
        digital = self.read_digital(start, count).astype(np.float64)
        baselines = np.asarray(
            [channel.baseline for channel in self.header.channels],
            dtype=np.float64,
        )[:, None]
        gains = np.asarray(
            [channel.gain for channel in self.header.channels],
            dtype=np.float64,
        )[:, None]
        return (digital - baselines) / gains

    def read(
        self,
        start: int,
        count: int,
        *,
        physical: bool = True,
    ) -> np.ndarray:
        """Read physical values by default, or native integers on request."""
        return (
            self.read_physical(start, count)
            if physical
            else self.read_digital(start, count)
        )

    def annotations_between(
        self,
        start: int,
        stop: int,
    ) -> tuple[WFDBAnnotation, ...]:
        """Return annotations in the half-open native sample interval."""
        if start < 0 or stop < start or stop > self.sample_count:
            raise ValueError("Annotation interval is outside the recording")
        return tuple(
            annotation
            for annotation in self.annotations
            if start <= annotation.sample < stop
        )

    def verify_signal_checksums(self) -> tuple[int, ...]:
        """Decode the complete record and validate WFDB signed checksums."""
        digital = self.read_digital(0, self.sample_count)
        sums = np.sum(digital, axis=1, dtype=np.int64) & 0xFFFF
        signed = np.where(sums >= 0x8000, sums - 0x10000, sums)
        actual = tuple(int(value) for value in signed)
        expected = tuple(channel.checksum for channel in self.header.channels)
        if actual != expected:
            raise ValueError(
                f"{self.data_path.name} checksum mismatch: "
                f"expected {expected}, decoded {actual}"
            )
        return actual

    def _read_format_212(self, start: int, count: int) -> np.ndarray:
        channels = self.header.channels
        if self.channel_count != 2:
            raise ValueError(
                "Exact ranged format-212 reads currently require two channels"
            )
        if any(
            channel.samples_per_frame != 1
            or channel.skew != 0
            or channel.byte_offset != 0
            for channel in channels
        ):
            raise ValueError(
                "Format-212 samples-per-frame, skew, or byte offsets are not supported"
            )
        with self.data_path.open("rb") as stream:
            stream.seek(start * 3)
            packed = np.fromfile(stream, dtype=np.uint8, count=count * 3)
        if packed.size != count * 3:
            raise ValueError(f"{self.data_path.name} ended during a ranged read")
        packed = packed.astype(np.int16, copy=False).reshape(-1, 3)
        first = packed[:, 0] + ((packed[:, 1] & 0x0F) << 8)
        second = packed[:, 2] + ((packed[:, 1] >> 4) << 8)
        first[first > 2047] -= 4096
        second[second > 2047] -= 4096
        return np.stack((first, second)).astype(np.int16, copy=False)

    def _read_format_16(self, start: int, count: int) -> np.ndarray:
        channels = self.header.channels
        if any(
            channel.samples_per_frame != 1
            or channel.skew != 0
            or channel.byte_offset != channels[0].byte_offset
            for channel in channels
        ):
            raise ValueError(
                "Format-16 samples-per-frame, skew, or mixed byte offsets "
                "are not supported"
            )
        frame_bytes = 2 * self.channel_count
        with self.data_path.open("rb") as stream:
            stream.seek(channels[0].byte_offset + start * frame_bytes)
            values = np.fromfile(
                stream,
                dtype="<i2",
                count=count * self.channel_count,
            )
        if values.size != count * self.channel_count:
            raise ValueError(f"{self.data_path.name} ended during a ranged read")
        return values.reshape(-1, self.channel_count).T.copy()


def _parse_format(token: str, path: Path) -> tuple[int, int, int, int]:
    match = _FORMAT_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError(f"{path.name} has an unsupported format token: {token}")
    return (
        int(match.group("format")),
        int(match.group("samples_per_frame") or 1),
        int(match.group("skew") or 0),
        int(match.group("byte_offset") or 0),
    )


def _parse_gain(
    token: str,
    adc_zero: int,
    path: Path,
) -> tuple[float, int, str]:
    match = _GAIN_PATTERN.fullmatch(token)
    if match is None or match.group("gain") is None:
        raise ValueError(f"{path.name} has an unsupported gain token: {token}")
    gain = float(match.group("gain"))
    if not isfinite(gain) or gain == 0:
        raise ValueError(f"{path.name} channel gain must be finite and nonzero")
    baseline = (
        int(match.group("baseline"))
        if match.group("baseline") is not None
        else adc_zero
    )
    return gain, baseline, match.group("units") or "mV"


def parse_wfdb_header(path: str | Path) -> WFDBHeader:
    """Parse a single-segment WFDB header used by common ECG records."""
    header_path = Path(path)
    lines = header_path.read_text(encoding="ascii").splitlines()
    content = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    comments = tuple(
        line.lstrip()[1:].strip() for line in lines if line.lstrip().startswith("#")
    )
    if not content:
        raise ValueError(f"{header_path.name} is empty")
    record_fields = content[0].split()
    if len(record_fields) < 4:
        raise ValueError(f"{header_path.name} has an incomplete record line")
    record_name = record_fields[0]
    if "/" in record_name:
        raise ValueError("Multi-segment WFDB records are not supported")
    channel_count = int(record_fields[1])
    sample_rate_token = record_fields[2].split("/", 1)[0]
    sample_rate = float(sample_rate_token)
    sample_count = int(record_fields[3])
    if (
        channel_count < 1
        or not isfinite(sample_rate)
        or sample_rate <= 0
        or sample_count < 1
    ):
        raise ValueError(f"{header_path.name} has invalid record dimensions")
    if len(content) != channel_count + 1:
        raise ValueError(
            f"{header_path.name} declares {channel_count} channels but "
            f"contains {len(content) - 1} signal lines"
        )

    channels: list[WFDBChannel] = []
    for line in content[1:]:
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"{header_path.name} has an incomplete signal line")
        file_name = fields[0]
        if Path(file_name).name != file_name:
            raise ValueError(f"{header_path.name} signal file must be a plain filename")
        data_format, samples_per_frame, skew, byte_offset = _parse_format(
            fields[1],
            header_path,
        )
        adc_resolution = int(fields[3])
        adc_zero = int(fields[4])
        gain, baseline, units = _parse_gain(
            fields[2],
            adc_zero,
            header_path,
        )
        channels.append(
            WFDBChannel(
                file_name=file_name,
                format=data_format,
                samples_per_frame=samples_per_frame,
                skew=skew,
                byte_offset=byte_offset,
                gain=gain,
                baseline=baseline,
                units=units,
                adc_resolution=adc_resolution,
                adc_zero=adc_zero,
                initial_value=int(fields[5]),
                checksum=int(fields[6]),
                block_size=int(fields[7]),
                name=" ".join(fields[8:]) or f"Channel {len(channels) + 1}",
            )
        )
    if len({channel.file_name for channel in channels}) != 1:
        raise ValueError("WFDB records split across signal files are not supported")
    return WFDBHeader(
        path=header_path,
        record_name=record_name,
        sample_rate=sample_rate,
        sample_count=sample_count,
        channels=tuple(channels),
        comments=comments,
    )


def load_wfdb_record(
    header_path: str | Path,
    *,
    annotation_extension: str = "atr",
) -> WFDBRecording:
    """Open one local WFDB record and its optional reference annotations."""
    path = Path(header_path)
    header = parse_wfdb_header(path)
    data_path = path.with_name(header.channels[0].file_name)
    if not data_path.is_file():
        raise ValueError(f"Missing WFDB signal data: {data_path.name}")
    annotation_path = path.with_suffix(f".{annotation_extension}")
    if annotation_path.is_file():
        annotations = read_mit_annotations(annotation_path)
    else:
        annotation_path = None
        annotations = ()

    expected_bytes: int | None
    formats = {channel.format for channel in header.channels}
    if formats == {212} and header.channel_count == 2:
        expected_bytes = header.sample_count * 3
    elif formats == {16}:
        expected_bytes = (
            header.channels[0].byte_offset
            + header.sample_count * header.channel_count * 2
        )
    else:
        expected_bytes = None
    if expected_bytes is not None and data_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"{data_path.name} size mismatch: expected {expected_bytes}, "
            f"found {data_path.stat().st_size}"
        )
    return WFDBRecording(
        header=header,
        data_path=data_path,
        annotation_path=annotation_path,
        annotations=annotations,
    )


__all__ = [
    "WFDBChannel",
    "WFDBHeader",
    "WFDBRecording",
    "load_wfdb_record",
    "parse_wfdb_header",
]
