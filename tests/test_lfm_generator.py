import unittest
from math import log10
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.generate_lfm_collection import (
    CALIBRATION_TONE_HZ,
    OTA_PRF_HZ,
    OTA_PULSE_WIDTH_SECONDS,
    OTA_TARGETS,
    PROFILES,
    R_OHMS,
    THERMAL_NOISE_DBM_HZ,
    VOLTS_PER_COUNT,
    generate_collection,
    noise_component_std,
    write_member,
)
from sigvue_examples.radar.domain import _averaged_psd, _single_psd


class LfmGeneratorTests(unittest.TestCase):
    def test_collection_manifest_uses_standard_streams_with_lfm_roles(self):
        with TemporaryDirectory() as directory, patch(
            "scripts.generate_lfm_collection.write_member"
        ):
            path = generate_collection(
                Path(directory),
                PROFILES["2mhz"],
                noise_figure_db=7.0,
                seed=1234,
            )
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            collection = payload["collection"]

        self.assertEqual("1.2.6", collection["core:version"])
        self.assertEqual("lfm", collection["core:extensions"][0]["name"])
        self.assertEqual(12, len(collection["core:streams"]))
        self.assertEqual(
            {"calibration", "terminated-noise", "ota"},
            {stream["lfm:role"] for stream in collection["core:streams"]},
        )
        self.assertNotIn("members", payload)

    def test_generator_exposes_original_and_multi_target_profiles(self):
        self.assertEqual({"10mhz", "10mhz-16ch", "2mhz"}, set(PROFILES))
        self.assertEqual(10_000_000, PROFILES["10mhz"].sample_rate)
        self.assertEqual(1, len(PROFILES["10mhz"].targets))
        self.assertEqual(16, PROFILES["10mhz-16ch"].channel_count)
        self.assertEqual(10_000_000, PROFILES["10mhz-16ch"].sample_rate)
        self.assertEqual(2_000_000, PROFILES["2mhz"].sample_rate)
        self.assertEqual(3, len(PROFILES["2mhz"].targets))

    def test_sixteen_channel_profile_writes_all_role_streams(self):
        with TemporaryDirectory() as directory, patch(
            "scripts.generate_lfm_collection.write_member"
        ) as write_member_mock:
            path = generate_collection(
                Path(directory), PROFILES["10mhz-16ch"],
                noise_figure_db=7.0, seed=1234,
            )
            import json

            streams = json.loads(path.read_text(encoding="utf-8"))["collection"]["core:streams"]

        self.assertEqual(48, len(streams))
        self.assertEqual(48, write_member_mock.call_count)
        self.assertEqual(set(range(1, 17)), {stream["lfm:channel"] for stream in streams})

    def test_noise_scale_encodes_requested_noise_figure(self):
        sample_rate = 2_000_000
        noise_figure = 7.0
        component_std = noise_component_std(sample_rate, noise_figure)
        complex_noise_power = (2 * component_std**2) / (2 * R_OHMS)
        psd_dbm_hz = 10 * log10((complex_noise_power / sample_rate) / 1e-3)
        self.assertAlmostEqual(THERMAL_NOISE_DBM_HZ + noise_figure, psd_dbm_hz)

    def test_terminated_noise_member_is_white_and_recovers_noise_figure(self):
        sample_rate = 2_000_000
        noise_figure = 7.0
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_member(
                root,
                "terminated-noise",
                channel=1,
                duration=0.02,
                sample_rate=sample_rate,
                calibration_dbm=-20.0,
                noise_figure_db=noise_figure,
                seed=1234,
            )
            iq = np.fromfile(root / "terminated-noise-ch1.sigmf-data", dtype="<i2").reshape(-1, 2)
            noise = (iq[:, 0].astype(float) + 1j * iq[:, 1].astype(float)) * VOLTS_PER_COUNT

        measured_power = np.mean(np.abs(noise) ** 2) / (2 * R_OHMS)
        measured_psd_dbm_hz = 10 * np.log10((measured_power / sample_rate) / 1e-3)
        lag_one_correlation = abs(np.vdot(noise[:-1], noise[1:])) / np.vdot(noise, noise).real
        _, averaged_psd = _averaged_psd(noise, sample_rate)
        # ci16 quantization is visible at this deliberately low 2 MHz noise floor.
        self.assertAlmostEqual(noise_figure, measured_psd_dbm_hz - THERMAL_NOISE_DBM_HZ, delta=0.25)
        self.assertLess(lag_one_correlation, 0.015)
        self.assertLess(float(np.std(averaged_psd)), 0.7)

    def test_contiguous_calibration_samples_retain_the_known_tone(self):
        sample_rate = 2_000_000
        time = np.arange(4096) / sample_rate
        tone = np.exp(1j * 2 * np.pi * CALIBRATION_TONE_HZ * time)
        frequency, psd = _single_psd(tone, sample_rate)
        self.assertLessEqual(abs(float(frequency[np.argmax(psd)]) - CALIBRATION_TONE_HZ), sample_rate / 1024)

    def test_generated_calibration_is_a_pure_known_power_tone(self):
        sample_rate = 2_000_000
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_member(root, "calibration", 1, 0.001, sample_rate, -20.0, 7.0, 1234)
            iq = np.fromfile(root / "calibration-ch1.sigmf-data", dtype="<i2").reshape(-1, 2)
        tone = (iq[:, 0].astype(float) + 1j * iq[:, 1].astype(float)) * VOLTS_PER_COUNT
        time = np.arange(tone.size) / sample_rate
        carrier = np.exp(1j * 2 * np.pi * CALIBRATION_TONE_HZ * time)
        coefficient = np.vdot(carrier, tone) / np.vdot(carrier, carrier)
        residual = tone - coefficient * carrier
        measured_dbm = 10 * np.log10((np.mean(np.abs(tone) ** 2) / (2 * R_OHMS)) / 1e-3)
        self.assertAlmostEqual(-20.0, measured_dbm, delta=0.01)
        self.assertLess(np.mean(np.abs(residual) ** 2) / np.mean(np.abs(tone) ** 2), 1e-8)

    def test_generated_ota_contains_three_delayed_lfm_returns(self):
        sample_rate = 2_000_000
        duration = 0.01
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_member(root, "ota", 1, duration, sample_rate, -20.0, 7.0, 1234)
            iq = np.fromfile(root / "ota-ch1.sigmf-data", dtype="<i2").reshape(-1, 2)
        ota = iq[:, 0].astype(float) + 1j * iq[:, 1].astype(float)
        pri_samples = round(sample_rate / OTA_PRF_HZ)
        pulse_samples = round(sample_rate * OTA_PULSE_WIDTH_SECONDS)
        rows = ota.reshape(round(duration * OTA_PRF_HZ), pri_samples)
        occupied = np.zeros(pri_samples, dtype=bool)
        target_powers = []
        for delay_seconds, _, _ in OTA_TARGETS:
            start = round(delay_seconds * sample_rate)
            occupied[start : start + pulse_samples] = True
            target_powers.append(np.mean(np.abs(rows[:, start : start + pulse_samples]) ** 2))
        off_power = np.mean(np.abs(rows[:, ~occupied]) ** 2)
        self.assertEqual(4, rows.shape[0])
        self.assertEqual(3, len(target_powers))
        self.assertTrue(all(10 * np.log10(power / off_power) > 45.0 for power in target_powers))
        self.assertGreater(target_powers[0], target_powers[1])
        self.assertGreater(target_powers[1], target_powers[2])


if __name__ == "__main__":
    unittest.main()
