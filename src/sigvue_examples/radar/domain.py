"""Domain models and numerical processing for calibrated LFM radar data."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log10, pi, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from sigvue.plugin import Annotation

R_OHMS = 50.0
THERMAL_NOISE_DBM_HZ = -174.0


@dataclass(frozen=True)
class CollectionMember:
    role: str
    channel: int
    metadata_path: Path
    data_path: Path
    duration: float


@dataclass(frozen=True)
class LfmCollection:
    sample_rate: float
    calibration_dbm: float
    adc_bits: int
    members: dict[str, tuple[CollectionMember, ...]]
    ota_prf_hz: float = 1_000.0
    ota_pulse_width_seconds: float = 50e-6
    collection_path: Path | None = None

    def sample_count(self, role: str) -> int:
        return min(member.data_path.stat().st_size // 4 for member in self.members[role])

    def read(self, role: str, start: int = 0, count: int | None = None) -> np.ndarray:
        available = self.sample_count(role)
        start = min(available, max(0, start))
        count = available - start if count is None else min(max(0, count), available - start)
        channels = []
        for member in self.members[role]:
            with member.data_path.open("rb") as stream:
                stream.seek(start * 4)
                iq = np.fromfile(stream, dtype="<i2", count=count * 2).reshape(-1, 2)
            channels.append(iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32))
        return np.asarray(channels, dtype=np.complex64)


@dataclass(frozen=True)
class LfmInput:
    sample_rate: float
    calibration_dbm: float
    adc_bits: int
    pri_samples: int
    start_sample: int
    calibration_counts: np.ndarray
    noise_counts: np.ndarray
    ota_counts: np.ndarray
    annotations: tuple[Annotation, ...] = ()


















@dataclass(frozen=True)
class Calibration:
    phase_offsets: np.ndarray
    volts_per_count: np.ndarray
    amplitude_corrections: np.ndarray
    reference_volts_per_count: float
    phase_reference_channel: int
    amplitude_reference_channel: int
    amplitude_reference_label: str
    noise_power_dbm: np.ndarray
    noise_psd_dbm_hz: np.ndarray
    noise_figure_db: np.ndarray
    full_scale_dbm: np.ndarray


@dataclass(frozen=True)
class Products:
    fast_time_us: np.ndarray
    slow_time_s: np.ndarray
    slow_time_edges_s: np.ndarray
    frequencies_hz: np.ndarray
    time_mean_dbm: np.ndarray
    time_max_dbm: np.ndarray
    time_waterfall_dbm: np.ndarray
    psd_mean_dbm_hz: np.ndarray
    psd_max_dbm_hz: np.ndarray
    psd_waterfall_dbm_hz: np.ndarray


@dataclass(frozen=True)
class LfmSettings:
    adc_bits: int
    phase_reference: str
    amplitude_reference: str
    reference_noise_psd_dbm_hz: float


@dataclass(frozen=True)
class LfmAnalysisProducts:
    data: LfmInput
    settings: LfmSettings
    calibration: Calibration
    signal: Products
    calibrated_tone: np.ndarray
    calibrated_noise: np.ndarray
    phase_rows: list[dict[str, object]]
    amplitude_rows: list[dict[str, object]]
    amplitude_summary: str
    noise_rows: list[dict[str, object]]




def process_lfm(data: LfmInput, settings: LfmSettings) -> LfmAnalysisProducts:
    calibration = _calibrate(
        data,
        adc_bits=settings.adc_bits,
        phase_reference=settings.phase_reference,
        amplitude_reference=settings.amplitude_reference,
    )
    ota = _apply_calibration(data.ota_counts, calibration)
    calibrated_tone = _apply_calibration(data.calibration_counts, calibration)
    calibrated_noise = data.noise_counts * calibration.volts_per_count[:, None]
    signal = _products(
        ota,
        data.sample_rate,
        data.pri_samples,
        data.start_sample,
    )
    phase_rows = [
        {
            "Channel": channel + 1,
            "Reference": "Yes" if channel == calibration.phase_reference_channel else "",
            "Phase correction": f"{-calibration.phase_offsets[channel] * 180 / pi:+.2f} deg",
        }
        for channel in range(4)
    ]
    amplitude_rows = [
        {
            "Channel": channel + 1,
            "Normalization": f"{calibration.amplitude_corrections[channel]:.4f}x",
            "Recorded full-scale power": f"{calibration.full_scale_dbm[channel]:.2f} dBm",
        }
        for channel in range(4)
    ]
    calibrated_full_scale_voltage = (2 ** (settings.adc_bits - 1) - 1) * calibration.reference_volts_per_count
    calibrated_full_scale_dbm = float(_db10((calibrated_full_scale_voltage**2 / (2 * R_OHMS)) / 1e-3))
    amplitude_summary = (
        f"Normalized to: **{calibration.amplitude_reference_label}**\n"
        f"Calibrated scale: **{calibration.reference_volts_per_count:.4g} V/count**\n"
        f"Calibrated full scale: **{calibrated_full_scale_dbm:.2f} dBm**"
    )
    noise_rows = [
        {
            "Channel": channel + 1,
            "Noise power": f"{calibration.noise_power_dbm[channel]:.2f} dBm",
            "Noise PSD": f"{calibration.noise_psd_dbm_hz[channel]:.2f} dBm/Hz",
            "Estimated NF": (
                f"{calibration.noise_psd_dbm_hz[channel] - settings.reference_noise_psd_dbm_hz:.2f} dB"
            ),
        }
        for channel in range(4)
    ]
    return LfmAnalysisProducts(
        data=data,
        settings=settings,
        calibration=calibration,
        signal=signal,
        calibrated_tone=calibrated_tone,
        calibrated_noise=calibrated_noise,
        phase_rows=phase_rows,
        amplitude_rows=amplitude_rows,
        amplitude_summary=amplitude_summary,
        noise_rows=noise_rows,
    )




def _calibrate(
    data: LfmInput,
    *,
    adc_bits: int | None = None,
    phase_reference: str = "Channel 1",
    amplitude_reference: str = "Min",
) -> Calibration:
    instantaneous_peak_power = np.max(np.abs(data.calibration_counts) ** 2, axis=1)
    phase_reference_channel = _reference_channel(phase_reference, instantaneous_peak_power, allow_min=False)
    amplitude_reference_channel = _reference_channel(amplitude_reference, instantaneous_peak_power, allow_min=True)
    reference = data.calibration_counts[phase_reference_channel]
    phase_offsets = np.asarray([np.angle(np.mean(channel * np.conj(reference))) for channel in data.calibration_counts])
    desired_voltage = sqrt(2 * R_OHMS * 1e-3 * 10 ** (data.calibration_dbm / 10))
    count_magnitude = np.sqrt(np.mean(np.abs(data.calibration_counts) ** 2, axis=1))
    peak_magnitude = np.sqrt(np.maximum(instantaneous_peak_power, 1e-24))
    amplitude_corrections = peak_magnitude[amplitude_reference_channel] / peak_magnitude
    reference_volts_per_count = desired_voltage / max(count_magnitude[amplitude_reference_channel], 1e-12)
    volts_per_count = amplitude_corrections * reference_volts_per_count
    noise_voltage = data.noise_counts * volts_per_count[:, None]
    noise_watts = np.mean(np.abs(noise_voltage) ** 2, axis=1) / (2 * R_OHMS)
    noise_power_dbm = _db10(noise_watts / 1e-3)
    noise_psd = noise_watts / data.sample_rate
    noise_psd_dbm_hz = _db10(noise_psd / 1e-3)
    noise_figure_db = noise_psd_dbm_hz - THERMAL_NOISE_DBM_HZ
    effective_adc_bits = data.adc_bits if adc_bits is None else adc_bits
    full_scale_voltage = (2 ** (effective_adc_bits - 1) - 1) * volts_per_count
    full_scale_dbm = _db10((full_scale_voltage**2 / (2 * R_OHMS)) / 1e-3)
    reference_label = (
        f"Min (Channel {amplitude_reference_channel + 1})"
        if amplitude_reference == "Min"
        else f"Channel {amplitude_reference_channel + 1}"
    )
    return Calibration(
        phase_offsets,
        volts_per_count,
        amplitude_corrections,
        reference_volts_per_count,
        phase_reference_channel,
        amplitude_reference_channel,
        reference_label,
        noise_power_dbm,
        noise_psd_dbm_hz,
        noise_figure_db,
        full_scale_dbm,
    )


def _reference_channel(value: str, peak_power: np.ndarray, *, allow_min: bool) -> int:
    if allow_min and value == "Min":
        return int(np.argmin(peak_power))
    if value.startswith("Channel "):
        try:
            index = int(value.removeprefix("Channel ")) - 1
        except ValueError:
            index = 0
        if 0 <= index < peak_power.size:
            return index
    return 0


def _apply_calibration(counts: np.ndarray, calibration: Calibration) -> np.ndarray:
    rotations = np.exp(-1j * calibration.phase_offsets).astype(np.complex64)
    normalized = counts * calibration.amplitude_corrections[:, None]
    return normalized * calibration.reference_volts_per_count * rotations[:, None]


def _products(
    channels: np.ndarray,
    rate: float,
    pri: int,
    start: int,
    max_rows: int | None = None,
    max_fast_time_bins: int | None = None,
    max_frequency_bins: int | None = None,
) -> Products:
    row_count = channels.shape[1] // pri
    if row_count < 1:
        raise ValueError("Delivered data must contain at least one PRI")
    rows = channels[:, : row_count * pri].reshape(4, row_count, pri)
    fast_group_size = 1 if max_fast_time_bins is None else max(1, ceil(pri / max_fast_time_bins))
    displayed_samples = pri // fast_group_size * fast_group_size
    fast_time_start = start % pri
    fast_time = (
        (fast_time_start + np.arange(0, displayed_samples, fast_group_size))
        / rate
        * 1e6
    )
    power = np.abs(rows) ** 2 / (2 * R_OHMS)
    mean_power = np.mean(power, axis=1)[:, :displayed_samples]
    max_power = np.max(power, axis=1)[:, :displayed_samples]
    time_mean = _db10(mean_power.reshape(4, -1, fast_group_size).mean(axis=2) / 1e-3)
    time_max = _db10(max_power.reshape(4, -1, fast_group_size).max(axis=2) / 1e-3)

    # Transform every sample in each PRI. Truncating the row here makes a
    # shifted pulse disappear from the PSD even though it remains visible in
    # the time waterfall. A rectangular full-row periodogram also preserves
    # spectral magnitude under circular shifts; display reduction happens only
    # after power has been calculated for every FFT bin.
    frequency_group_size = (
        1 if max_frequency_bins is None else max(1, ceil(pri / max_frequency_bins))
    )
    full_frequencies = np.fft.fftshift(np.fft.fftfreq(pri, d=1 / rate))
    frequencies = _group_mean(full_frequencies, frequency_group_size)
    frequency_bin_hz = rate / pri
    psd_sum = np.zeros((4, frequencies.size), dtype=np.float64)
    psd_max = np.zeros((4, frequencies.size), dtype=np.float64)
    time_waterfall = []
    psd_waterfall = []
    slow_time = []
    slow_time_edges = [0.0]
    group_size = 1 if max_rows is None else max(1, ceil(row_count / max_rows))
    for first in range(0, row_count, group_size):
        block = rows[:, first : min(first + group_size, row_count)]
        block_power = np.abs(block) ** 2 / (2 * R_OHMS)
        waterfall_power = np.mean(block_power, axis=1)[:, :displayed_samples]
        waterfall_power = waterfall_power.reshape(4, -1, fast_group_size).mean(axis=2)
        time_waterfall.append(_db10(waterfall_power / 1e-3))
        spectrum = np.fft.fftshift(np.fft.fft(block, axis=2), axes=2)
        full_psd = np.abs(spectrum) ** 2 / pri**2 / (2 * R_OHMS) / frequency_bin_hz
        psd = _group_mean(full_psd, frequency_group_size)
        psd_sum += np.sum(psd, axis=1)
        psd_max = np.maximum(psd_max, np.max(psd, axis=1))
        psd_waterfall.append(_db10(np.mean(psd, axis=1) / 1e-3))
        slow_time.append((first + block.shape[1] / 2) * pri / rate)
        slow_time_edges.append((first + block.shape[1]) * pri / rate)
    psd_mean = _db10((psd_sum / row_count) / 1e-3)
    psd_hold = _db10(psd_max / 1e-3)
    return Products(
        fast_time,
        np.asarray(slow_time),
        np.asarray(slow_time_edges),
        frequencies,
        time_mean,
        time_max,
        np.stack(time_waterfall, axis=1),
        psd_mean,
        psd_hold,
        np.stack(psd_waterfall, axis=1),
    )


def _group_mean(values: np.ndarray, group_size: int) -> np.ndarray:
    """Average adjacent values on the final axis without dropping a tail."""
    if group_size <= 1:
        return values
    starts = np.arange(0, values.shape[-1], group_size)
    counts = np.diff(np.append(starts, values.shape[-1]))
    shape = (1,) * (values.ndim - 1) + (counts.size,)
    return np.add.reduceat(values, starts, axis=-1) / counts.reshape(shape)


def _db10(value: Any) -> np.ndarray:
    return 10 * np.log10(np.maximum(value, 1e-30))










def _single_psd(samples: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    nfft = min(1024, samples.size)
    window = np.hanning(nfft)
    spectrum = np.fft.fftshift(np.fft.fft(samples[:nfft] * window))
    psd = np.abs(spectrum) ** 2 / (rate * np.sum(window**2) * 2 * R_OHMS)
    return np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / rate)), _db10(psd / 1e-3)


def _averaged_psd(
    samples: np.ndarray,
    rate: float,
    *,
    nfft: int = 1024,
    max_blocks: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    nfft = min(nfft, samples.size)
    block_count = samples.size // nfft
    if block_count < 1:
        return _single_psd(samples, rate)
    stride = max(1, block_count // max_blocks)
    blocks = samples[: block_count * nfft].reshape(block_count, nfft)[::stride][:max_blocks]
    window = np.hanning(nfft)
    spectra = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    psd = np.mean(np.abs(spectra) ** 2, axis=0) / (rate * np.sum(window**2) * 2 * R_OHMS)
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / rate))
    return frequencies, _db10(psd / 1e-3)
