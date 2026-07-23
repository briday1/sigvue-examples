"""Generate standard SigMF collections for the calibrated LFM workspace."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import pi
from pathlib import Path

import numpy as np


R_OHMS = 50.0
THERMAL_NOISE_DBM_HZ = -174.0
DEFAULT_NOISE_FIGURE_DB = 7.0
VOLTS_PER_COUNT = 1e-6

# Defaults retained for direct write_member() use and focused generator tests.
CALIBRATION_TONE_HZ = 200_000.0
OTA_PRF_HZ = 400.0
OTA_PULSE_WIDTH_SECONDS = 160e-6
OTA_SWEEP_BANDWIDTH_HZ = 1_200_000.0
OTA_TARGETS = (
    (40e-6, 0.62, 0.0),
    (350e-6, 0.34, 120.0),
    (900e-6, 0.20, -180.0),
)


@dataclass(frozen=True)
class LfmProfile:
    key: str
    name: str
    sample_rate: int
    calibration_tone_hz: float
    prf_hz: float
    pulse_width_seconds: float
    sweep_bandwidth_hz: float
    targets: tuple[tuple[float, float, float], ...]
    channel_count: int = 4


PROFILES = {
    "10mhz": LfmProfile(
        "10mhz",
        "Synthetic 10 MHz four-channel LFM collection",
        10_000_000,
        1_000_000.0,
        1_000.0,
        50e-6,
        4_000_000.0,
        ((0.0, 1.0, 0.0),),
    ),
    "10mhz-16ch": LfmProfile(
        "10mhz-16ch",
        "Synthetic 10 MHz sixteen-channel LFM collection",
        10_000_000,
        1_000_000.0,
        1_000.0,
        50e-6,
        4_000_000.0,
        ((0.0, 1.0, 0.0),),
        16,
    ),
    "2mhz": LfmProfile(
        "2mhz",
        "Synthetic 2 MHz multi-target four-channel LFM collection",
        2_000_000,
        CALIBRATION_TONE_HZ,
        OTA_PRF_HZ,
        OTA_PULSE_WIDTH_SECONDS,
        OTA_SWEEP_BANDWIDTH_HZ,
        OTA_TARGETS,
    ),
}


def noise_component_std(sample_rate: float, noise_figure_db: float) -> float:
    """Return each I/Q component's voltage standard deviation for complex noise."""
    noise_psd_watts_hz = 1e-3 * 10 ** ((THERMAL_NOISE_DBM_HZ + noise_figure_db) / 10)
    noise_power_watts = noise_psd_watts_hz * sample_rate
    return np.sqrt(noise_power_watts * R_OHMS)


def write_member(
    root: Path,
    role: str,
    channel: int,
    duration: float,
    sample_rate: int,
    calibration_dbm: float,
    noise_figure_db: float,
    seed: int,
    *,
    calibration_tone_hz: float = CALIBRATION_TONE_HZ,
    ota_prf_hz: float = OTA_PRF_HZ,
    ota_pulse_width_seconds: float = OTA_PULSE_WIDTH_SECONDS,
    ota_sweep_bandwidth_hz: float = OTA_SWEEP_BANDWIDTH_HZ,
    ota_targets: tuple[tuple[float, float, float], ...] = OTA_TARGETS,
) -> None:
    count = round(duration * sample_rate)
    metadata = root / f"{role}-ch{channel}.sigmf-meta"
    data = root / f"{role}-ch{channel}.sigmf-data"
    metadata.write_text(
        json.dumps(
            {
                "global": {
                    "core:datatype": "ci16_le",
                    "core:sample_rate": sample_rate,
                    "core:num_channels": 1,
                    "core:description": f"Synthetic {role}, channel {channel}; interleaved IQ",
                }
            }
        ),
        encoding="utf-8",
    )
    amplitude = np.sqrt(2 * R_OHMS * 1e-3 * 10 ** (calibration_dbm / 10))
    noise_std = noise_component_std(sample_rate, noise_figure_db)
    generator = np.random.default_rng(seed)
    calibration_phases = (0.0, 0.37, -0.68, 1.04)
    phase = (
        calibration_phases[channel - 1]
        if channel <= len(calibration_phases)
        else ((channel - 1) * 0.61803398875 * 2 * pi) % (2 * pi) - pi
    )
    chunk = 250_000
    with data.open("wb") as stream:
        for start in range(0, count, chunk):
            index = np.arange(start, min(start + chunk, count))
            time = index / sample_rate
            if role == "calibration":
                signal = amplitude * np.exp(1j * (2 * pi * calibration_tone_hz * time + phase))
            else:
                noise = noise_std * (
                    generator.standard_normal(time.size) + 1j * generator.standard_normal(time.size)
                )
                if role == "terminated-noise":
                    signal = noise
                else:
                    pri_samples = round(sample_rate / ota_prf_hz)
                    pulse_samples = round(sample_rate * ota_pulse_width_seconds)
                    fast_sample = np.remainder(index, pri_samples)
                    chirp_rate = ota_sweep_bandwidth_hz / ota_pulse_width_seconds
                    signal = noise.astype(np.complex128)
                    for target_index, (delay_seconds, relative_amplitude, doppler_hz) in enumerate(ota_targets):
                        delay_samples = round(delay_seconds * sample_rate)
                        target_sample = fast_sample - delay_samples
                        target_time = target_sample / sample_rate
                        pulse_active = (target_sample >= 0) & (target_sample < pulse_samples)
                        chirp_phase = 2 * pi * (
                            -ota_sweep_bandwidth_hz / 2 * target_time
                            + 0.5 * chirp_rate * target_time**2
                            + doppler_hz * time
                        ) + phase + target_index * 0.31
                        signal += pulse_active * relative_amplitude * amplitude * np.exp(1j * chirp_phase)
            iq = np.empty((len(time), 2), dtype="<i2")
            iq[..., 0] = np.clip(np.rint(signal.real / VOLTS_PER_COUNT), -32768, 32767)
            iq[..., 1] = np.clip(np.rint(signal.imag / VOLTS_PER_COUNT), -32768, 32767)
            iq.tofile(stream)


def generate_collection(root: Path, profile: LfmProfile, noise_figure_db: float, seed: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for path in root.glob("*.sigmf-*"):
        path.unlink()
    power = -20.0
    streams = []
    for role_index, (role, duration) in enumerate(
        (("calibration", 0.1), ("terminated-noise", 0.1), ("ota", 1.0))
    ):
        for channel in range(1, profile.channel_count + 1):
            write_member(
                root,
                role,
                channel,
                duration,
                profile.sample_rate,
                power,
                noise_figure_db,
                seed + role_index * 100 + channel,
                calibration_tone_hz=profile.calibration_tone_hz,
                ota_prf_hz=profile.prf_hz,
                ota_pulse_width_seconds=profile.pulse_width_seconds,
                ota_sweep_bandwidth_hz=profile.sweep_bandwidth_hz,
                ota_targets=profile.targets,
            )
            streams.append(
                {
                    "name": f"{role}-ch{channel}.sigmf-meta",
                    "lfm:role": role,
                    "lfm:channel": channel,
                    "lfm:duration_seconds": duration,
                }
            )
    manifest = {
        "collection": {
            "core:version": "1.2.6",
            "core:description": profile.name,
            "core:extensions": [{"name": "lfm", "version": "0.1.0", "optional": False}],
            "core:streams": streams,
            "lfm:calibration_dbm": power,
            "lfm:ota_prf_hz": profile.prf_hz,
            "lfm:ota_pulse_width_seconds": profile.pulse_width_seconds,
        },
    }
    manifest_path = root / f"lfm-{profile.key}.sigmf-collection"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Generated {len(streams)} streams for {profile.name} in {root} "
        f"with hidden true noise figure {noise_figure_db:g} dB"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/lfm-sigmf"), help="parent directory for profile data")
    parser.add_argument("--profile", choices=("all", *PROFILES), default="all")
    parser.add_argument("--noise-figure-db", type=float, default=DEFAULT_NOISE_FIGURE_DB)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    selected = PROFILES.values() if args.profile == "all" else (PROFILES[args.profile],)
    for profile in selected:
        generate_collection(
            (args.output / profile.key).resolve(),
            profile,
            args.noise_figure_db,
            args.seed,
        )


if __name__ == "__main__":
    main()
