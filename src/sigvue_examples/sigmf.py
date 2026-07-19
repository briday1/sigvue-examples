"""Framework-independent SigMF metadata and ranged sample I/O."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import RLock

import numpy as np


_metadata_lock = RLock()


@dataclass(frozen=True)
class SigMFRecording:
    metadata_path: Path
    data_path: Path
    sample_rate: float
    channel_count: int
    sample_count: int
    metadata: dict[str, object]
    datatype: str

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate

    def read(self, start: int, count: int) -> np.ndarray:
        """Read only the requested interleaved complex frames."""
        start = min(self.sample_count, max(0, start))
        count = min(max(0, count), self.sample_count - start)
        scalar_type, scalar_bytes, scale = {
            "cf32_le": ("<f4", 4, 1.0),
            "ci16_le": ("<i2", 2, 1.0 / 32768.0),
        }[self.datatype]
        scalars_per_frame = self.channel_count * 2
        with self.data_path.open("rb") as stream:
            stream.seek(start * scalars_per_frame * scalar_bytes)
            scalars = np.fromfile(stream, dtype=scalar_type, count=count * scalars_per_frame)
        frames = scalars.reshape(-1, self.channel_count, 2)
        return np.asarray((frames[..., 0] + 1j * frames[..., 1]) * scale, dtype=np.complex64).T


def load_metadata(metadata_path: Path) -> dict[str, object]:
    """Read a SigMF metadata document without applying workspace semantics."""
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def annotations(metadata_path: Path) -> tuple[dict[str, object], ...]:
    """Return current on-disk SigMF annotations without caching workspace state."""
    return tuple(load_metadata(metadata_path).get("annotations", ()))


def append_annotation(metadata_path: Path, annotation: dict[str, object]) -> None:
    """Atomically append and sample-sort one standard SigMF annotation object."""
    with _metadata_lock:
        metadata = load_metadata(metadata_path)
        entries = list(metadata.get("annotations", ()))
        entries.append(dict(annotation))
        entries.sort(key=lambda entry: int(entry["core:sample_start"]))
        metadata["annotations"] = entries
        temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(metadata_path)


def load_recording(
    metadata_path: Path,
    *,
    sample_rate_fallback: float | None = None,
) -> SigMFRecording:
    metadata = load_metadata(metadata_path)
    global_metadata = metadata["global"]
    datatype = str(global_metadata.get("core:datatype"))
    scalar_bytes = {"cf32_le": 4, "ci16_le": 2}.get(datatype)
    if scalar_bytes is None:
        raise ValueError(f"Unsupported SigMF datatype: {datatype}")
    channel_count = int(global_metadata.get("core:num_channels", 1))
    raw_sample_rate = global_metadata.get("core:sample_rate")
    if raw_sample_rate is None and sample_rate_fallback is None:
        raise ValueError(f"{metadata_path.name} does not define core:sample_rate")
    sample_rate = float(raw_sample_rate if raw_sample_rate is not None else sample_rate_fallback)
    if sample_rate <= 0:
        raise ValueError(f"{metadata_path.name} must have a positive sample rate")
    data_path = metadata_path.with_name(metadata_path.name.removesuffix(".sigmf-meta") + ".sigmf-data")
    sample_count = data_path.stat().st_size // (channel_count * 2 * scalar_bytes)
    return SigMFRecording(
        metadata_path,
        data_path,
        sample_rate,
        channel_count,
        sample_count,
        metadata,
        datatype,
    )
