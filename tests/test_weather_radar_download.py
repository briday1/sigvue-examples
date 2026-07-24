from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import call, patch

from sigvue.helpers import RemoteFile

from scripts import download_weather_radar


class WeatherRadarDownloadTests(unittest.TestCase):
    def test_manifest_pins_two_dense_two_hour_sequences(self):
        self.assertEqual(("TLX", "FDR"), download_weather_radar.DEFAULT_RADARS)
        self.assertEqual(
            {"TLX": 72, "FDR": 69},
            {
                radar: len(manifest)
                for radar, manifest in (
                    download_weather_radar.WEATHER_RADAR_SEQUENCES.items()
                )
            },
        )
        self.assertEqual(141, len(download_weather_radar.WEATHER_RADAR_FILES))
        self.assertEqual(
            (
                "TLX_N0B_2024_05_20_03_01_13",
                "TLX_N0B_2024_05_20_04_59_27",
                "FDR_N0B_2024_05_20_03_01_14",
                "FDR_N0B_2024_05_20_04_58_10",
            ),
            (
                download_weather_radar.WEATHER_RADAR_SEQUENCES["TLX"][0].filename,
                download_weather_radar.WEATHER_RADAR_SEQUENCES["TLX"][-1].filename,
                download_weather_radar.WEATHER_RADAR_SEQUENCES["FDR"][0].filename,
                download_weather_radar.WEATHER_RADAR_SEQUENCES["FDR"][-1].filename,
            ),
        )
        for remote in download_weather_radar.WEATHER_RADAR_FILES:
            self.assertIsInstance(remote, RemoteFile)
            self.assertGreater(remote.size, 0)
            self.assertRegex(remote.checksum, r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(
                f"{download_weather_radar.NEXRAD_BASE_URL}/{remote.filename}",
                remote.url,
            )

    def test_downloader_materializes_every_scan(self):
        progress = object()
        with (
            TemporaryDirectory() as directory,
            patch.object(
                download_weather_radar,
                "download_file",
            ) as download,
            patch.object(
                download_weather_radar,
                "_progress",
                return_value=progress,
            ) as progress_factory,
        ):
            output = Path(directory) / "weather-radar"
            expected = tuple(
                output / remote.filename
                for remote in download_weather_radar.WEATHER_RADAR_FILES
            )
            download.side_effect = expected
            result = download_weather_radar.download_weather_radar_scans(
                output,
                workers=1,
            )

        self.assertEqual(expected, result)
        self.assertEqual(
            [
                call(
                    remote,
                    output,
                    user_agent=download_weather_radar.USER_AGENT,
                    progress=progress,
                )
                for remote in download_weather_radar.WEATHER_RADAR_FILES
            ],
            download.call_args_list,
        )
        self.assertEqual(
            [
                call(remote.filename)
                for remote in download_weather_radar.WEATHER_RADAR_FILES
            ],
            progress_factory.call_args_list,
        )

    def test_downloader_can_select_one_radar_sequence(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(
                download_weather_radar,
                "_download_scan",
                side_effect=lambda remote, output: Path(output) / remote.filename,
            ) as download,
        ):
            output = Path(directory)
            paths = download_weather_radar.download_weather_radar_scans(
                output,
                ("FDR",),
                workers=1,
            )

        self.assertEqual(69, len(paths))
        self.assertTrue(all(path.name.startswith("FDR_") for path in paths))
        self.assertEqual(69, download.call_count)

    def test_downloader_rejects_invalid_worker_count(self):
        with self.assertRaisesRegex(ValueError, "workers"):
            download_weather_radar.download_weather_radar_scans(
                "unused",
                workers=0,
            )

    def test_parallel_downloader_preserves_manifest_order(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(
                download_weather_radar,
                "_download_scan",
                side_effect=lambda remote, output: Path(output) / remote.filename,
            ),
        ):
            paths = download_weather_radar.download_weather_radar_scans(
                directory,
                ("TLX",),
                workers=4,
            )

        self.assertEqual(
            tuple(
                Path(directory) / remote.filename
                for remote in download_weather_radar.WEATHER_RADAR_SEQUENCES["TLX"]
            ),
            paths,
        )


if __name__ == "__main__":
    unittest.main()
