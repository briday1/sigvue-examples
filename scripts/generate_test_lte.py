"""Generate compact deterministic ci16 LTE-shaped fixtures for automated tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RECORDINGS = (
    ("downlink", "LTE_downlink_806MHz_2022-04-09_30720ksps", 806_000_000.0, 20220409),
    ("uplink", "LTE_uplink_847MHz_2022-01-30_30720ksps", 847_000_000.0, 20220130),
)


def generate(root: Path) -> tuple[Path, ...]:
    sample_rate = 30_720_000.0
    count = 700_000
    time = np.arange(count) / sample_rate
    written = []
    for direction, name, center_hz, seed in RECORDINGS:
        rng = np.random.default_rng(seed)
        signal = 0.42 * np.exp(2j * np.pi * 2_100_000 * time)
        signal += 0.18 * np.exp(-2j * np.pi * 5_300_000 * time)
        signal += 0.025 * (rng.normal(size=count) + 1j * rng.normal(size=count))
        iq = np.empty((count, 2), dtype="<i2")
        iq[:, 0] = np.clip(np.rint(signal.real * 32767), -32768, 32767)
        iq[:, 1] = np.clip(np.rint(signal.imag * 32767), -32768, 32767)
        destination = root / direction
        destination.mkdir(parents=True, exist_ok=True)
        data_path = destination / f"{name}.sigmf-data"
        metadata_path = destination / f"{name}.sigmf-meta"
        iq.tofile(data_path)
        metadata_path.write_text(json.dumps({
            "global": {
                "core:datatype": "ci16_le",
                "core:sample_rate": sample_rate,
                "core:num_channels": 1,
                "core:description": f"Compact deterministic LTE {direction} test fixture",
            },
            "captures": [{"core:sample_start": 0, "core:frequency": center_hz}],
            "annotations": [],
        }, indent=2) + "\n", encoding="utf-8")
        written.extend((metadata_path, data_path))
    return tuple(written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in generate(args.output):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
