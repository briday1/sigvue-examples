"""Generate small deterministic communications recordings in ignored local data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_sigmf(root: Path, name: str, samples: np.ndarray, sample_rate: float, description: str, **extra) -> None:
    root.mkdir(parents=True, exist_ok=True)
    channels = 1 if samples.ndim == 1 else samples.shape[0]
    frames = samples.reshape(1, -1).T if channels == 1 else samples.T
    np.asarray(frames, dtype="<c8").tofile(root / f"{name}.sigmf-data")
    metadata = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": sample_rate,
            "core:num_channels": channels,
            "core:description": description,
            **extra,
        },
        "captures": [{"core:sample_start": 0}],
        "annotations": [],
    }
    (root / f"{name}.sigmf-meta").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def modulated_recording(
    modulation: str,
    *,
    sample_rate: float = 100_000.0,
    duration: float = 1.0,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = round(sample_rate * duration)
    samples_per_symbol = 10
    symbol_count = count // samples_per_symbol + 1
    if modulation == "QPSK":
        levels_i = 2 * rng.integers(0, 2, symbol_count) - 1
        levels_q = 2 * rng.integers(0, 2, symbol_count) - 1
        symbols = (levels_i + 1j * levels_q) / np.sqrt(2)
        gain = 0.72
    elif modulation == "16-QAM":
        levels = np.asarray([-3, -1, 1, 3])
        symbols = (rng.choice(levels, symbol_count) + 1j * rng.choice(levels, symbol_count)) / np.sqrt(10)
        gain = 0.68
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")
    baseband = np.repeat(symbols, samples_per_symbol)[:count]
    time = np.arange(count) / sample_rate
    envelope = gain * (0.94 + 0.06 * np.sin(2 * np.pi * 1.5 * time))
    noise_level = 0.018 + 0.035 * ((time > 0.72) & (time < 0.9))
    noise = noise_level * (rng.normal(size=count) + 1j * rng.normal(size=count))
    return np.asarray(envelope * baseband * np.exp(1j * 2 * np.pi * 7_000 * time) + noise, dtype=np.complex64)


def qpsk(sample_rate: float = 100_000.0, duration: float = 1.0) -> np.ndarray:
    return modulated_recording("QPSK", sample_rate=sample_rate, duration=duration, seed=10)


def qam16(sample_rate: float = 100_000.0, duration: float = 1.0) -> np.ndarray:
    return modulated_recording("16-QAM", sample_rate=sample_rate, duration=duration, seed=16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    write_sigmf(
        args.output / "comms",
        "qpsk",
        qpsk(),
        100_000.0,
        "Synthetic QPSK recording",
        **{
            "examples:modulation": "QPSK",
            "examples:symbol_rate": 10_000.0,
            "examples:carrier_hz": 7_000.0,
            "examples:constellation_limit": 0.8,
            "examples:eye_limit": 0.9,
        },
    )
    write_sigmf(
        args.output / "comms",
        "16qam",
        qam16(),
        100_000.0,
        "Synthetic 16-QAM recording",
        **{
            "examples:modulation": "16-QAM",
            "examples:symbol_rate": 10_000.0,
            "examples:carrier_hz": 7_000.0,
            "examples:constellation_limit": 0.8,
            "examples:eye_limit": 0.9,
        },
    )


if __name__ == "__main__":
    main()
