"""Generate the two small deterministic SigMF recordings committed with this repository."""

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


def qpsk(sample_rate: float = 100_000.0, duration: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(10)
    count = round(sample_rate * duration)
    samples_per_symbol = 10
    symbols = ((2 * rng.integers(0, 2, count // samples_per_symbol + 1) - 1) + 1j * (2 * rng.integers(0, 2, count // samples_per_symbol + 1) - 1)) / np.sqrt(2)
    baseband = np.repeat(symbols, samples_per_symbol)[:count]
    time = np.arange(count) / sample_rate
    envelope = 0.35 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * time))
    envelope += 0.18 * ((time > 0.42) & (time < 0.62))
    noise_level = 0.025 + 0.055 * ((time > 0.72) & (time < 0.9))
    noise = noise_level * (rng.normal(size=count) + 1j * rng.normal(size=count))
    return np.asarray(envelope * baseband * np.exp(1j * 2 * np.pi * 7_000 * time) + noise, dtype=np.complex64)


def multiple_tones(sample_rate: float = 100_000.0, duration: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(20)
    count = round(sample_rate * duration)
    time = np.arange(count) / sample_rate
    samples = 0.02 * (rng.normal(size=count) + 1j * rng.normal(size=count))
    definitions = (
        (-28_000.0, 0.24, np.ones(count, dtype=float)),
        (-11_000.0, 0.42, ((time >= 0.15) & (time < 0.85)).astype(float)),
        (6_000.0, 0.34, ((time >= 0.65) & (time < 1.55)).astype(float)),
        (23_000.0, 0.48, ((np.floor(time / 0.16) % 2) == 0).astype(float)),
    )
    for frequency, amplitude, gate in definitions:
        samples += amplitude * gate * np.exp(1j * 2 * np.pi * frequency * time)
    return np.asarray(samples, dtype=np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    write_sigmf(
        args.output,
        "qpsk",
        qpsk(),
        100_000.0,
        "Synthetic QPSK recording",
        **{"examples:symbol_rate": 10_000.0, "examples:carrier_hz": 7_000.0},
    )
    write_sigmf(
        args.output,
        "multiple-tones",
        multiple_tones(),
        100_000.0,
        "Time-varying multi-tone recording",
        **{"examples:tone_frequencies_hz": [-28_000.0, -11_000.0, 6_000.0, 23_000.0]},
    )


if __name__ == "__main__":
    main()
