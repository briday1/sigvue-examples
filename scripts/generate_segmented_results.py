"""Generate irregular precomputed acoustic-event display products."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


EVENTS = (
    ("event-001", 3.2, 0.42, "Door close", 0.97, 180.0),
    ("event-002", 11.7, 0.75, "Compressor start", 0.91, 420.0),
    ("event-003", 19.1, 0.31, "Metal impact", 0.95, 980.0),
    ("event-004", 37.8, 1.10, "Fan imbalance", 0.88, 235.0),
    ("event-005", 52.4, 0.58, "Valve actuation", 0.93, 610.0),
    ("event-006", 54.0, 0.24, "Metal impact", 0.89, 1_260.0),
    ("event-007", 81.6, 0.86, "Compressor stop", 0.94, 360.0),
)


def generate(path: Path) -> Path:
    rng = np.random.default_rng(30)
    results = []
    for identifier, start, duration, label, confidence, frequency in EVENTS:
        sample_rate = 8_000.0
        sample_count = max(64, round(duration * sample_rate))
        time = np.arange(sample_count) / sample_rate
        envelope = np.exp(-3.5 * time / duration)
        raw = envelope * np.sin(2 * np.pi * frequency * time)
        raw += 0.22 * envelope * np.sin(2 * np.pi * frequency * 2.15 * time + 0.4)
        raw += 0.025 * rng.normal(size=sample_count)

        display_indices = np.linspace(0, sample_count - 1, 400).astype(int)
        taper = np.hanning(sample_count)
        spectrum = np.abs(np.fft.rfft(raw * taper))
        spectrum_db = 20 * np.log10(np.maximum(spectrum / max(np.max(spectrum), 1e-12), 1e-4))
        spectrum_frequency = np.fft.rfftfreq(sample_count, 1 / sample_rate)
        spectrum_indices = np.linspace(0, spectrum_frequency.size - 1, 256).astype(int)

        results.append({
            "id": identifier,
            "label": label,
            "start_seconds": start,
            "duration_seconds": duration,
            "confidence": confidence,
            "waveform_time": time[display_indices].round(7).tolist(),
            "waveform": raw[display_indices].round(7).tolist(),
            "spectrum_frequency": spectrum_frequency[spectrum_indices].round(3).tolist(),
            "spectrum_db": spectrum_db[spectrum_indices].round(4).tolist(),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "title": "Synthetic acoustic event results",
        "duration_seconds": 90.0,
        "processing": "Precomputed and decimated waveform/spectrum display products",
        "events": results,
    }, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    output = generate(Path(__file__).resolve().parents[1] / "data" / "acoustic-events.json")
    print(f"Wrote {output}")
