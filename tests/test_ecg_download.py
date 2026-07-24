from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sigvue.helpers import RemoteFile

from scripts import download_mit_bih_ecg


class ECGDownloadTests(unittest.TestCase):
    def test_manifests_are_pinned_to_official_physionet_files(self):
        expected = {
            "100": (
                (
                    143,
                    "60ebc904c7bf3e04d142638d3fd5c903e8dc9f10f1ea3264e07926aa089ee75e",
                ),
                (
                    1_950_000,
                    "b2ea3c250e56e48f4b7b90697832b8ecd1afa1e0bb31f2dcfea4ed6e1075a639",
                ),
                (
                    4_558,
                    "8d8a5349fb16638ebbf649f1779d12e96d91b736b2aafe59db43719ae583d471",
                ),
            ),
            "101": (
                (
                    131,
                    "d5f02fbe8673fa05465442191b98ca0d28a1670e7ef0e83fb9ef8723113a311c",
                ),
                (
                    1_950_000,
                    "698d1ea6f472d23ca50317c72c96cda2698badd8578220ed0380cdf241e39006",
                ),
                (
                    3_768,
                    "441cdd6486cfdf4c53e344d1048ab81296773e7058d53069022adc68543a0663",
                ),
            ),
            "200": (
                (
                    306,
                    "9e0c2ff5b790cf624deab0ccb8a9f211a9e29a748d8197da3c1ee7c5b596b40c",
                ),
                (
                    1_950_000,
                    "a9e203b3807b9fcd3647cde03444437cb8eec7f5128a8eb413edafb394272e0f",
                ),
                (
                    8_114,
                    "f9624a11696760427d75314a78c31f89ca3c446af855890c8f8b66cddd8b3a3f",
                ),
            ),
            "207": (
                (
                    546,
                    "7645d488d4c304760aae0a709193ffa13692c317b551bcbdcbb37011032178d8",
                ),
                (
                    1_950_000,
                    "139f99250366fbf347cba4d8ea1fbe788f98cb1b93b70f50ccba70122d908605",
                ),
                (
                    4_958,
                    "cceb64d68033a277d2d5669458d49258295102dd3e20614a0ca7d63b67009404",
                ),
            ),
        }

        self.assertEqual(tuple(expected), download_mit_bih_ecg.DEFAULT_RECORDS)
        for record, values in expected.items():
            manifest = download_mit_bih_ecg.MIT_BIH_RECORDS[record]
            self.assertEqual(
                tuple(f"{record}.{extension}" for extension in ("hea", "dat", "atr")),
                tuple(item.filename for item in manifest),
            )
            self.assertTrue(all(isinstance(item, RemoteFile) for item in manifest))
            for item, (size, digest) in zip(manifest, values, strict=True):
                self.assertEqual(
                    f"https://physionet.org/files/mitdb/1.0.0/{item.filename}",
                    item.url,
                )
                self.assertEqual(size, item.size)
                self.assertEqual(f"sha256:{digest}", item.checksum)

    def test_downloads_selected_records_without_network(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "ecg"
            manifests = (
                download_mit_bih_ecg.MIT_BIH_RECORDS["101"],
                download_mit_bih_ecg.MIT_BIH_RECORDS["207"],
            )
            expected_by_record = tuple(
                tuple(destination / remote.filename for remote in manifest)
                for manifest in manifests
            )
            with patch.object(
                download_mit_bih_ecg,
                "download_wfdb_record",
                side_effect=expected_by_record,
            ) as download:
                paths = download_mit_bih_ecg.download_mit_bih_records(
                    destination,
                    ("101", "207"),
                )

        self.assertEqual(
            tuple(path for group in expected_by_record for path in group),
            paths,
        )
        self.assertEqual(2, download.call_count)
        for call, manifest in zip(download.call_args_list, manifests, strict=True):
            self.assertEqual((manifest, destination), call.args)
            self.assertEqual(
                {
                    "user_agent": download_mit_bih_ecg.USER_AGENT,
                    "progress_factory": download_mit_bih_ecg._progress,
                },
                call.kwargs,
            )

    def test_singular_download_defaults_to_record_100(self):
        with (
            TemporaryDirectory() as directory,
            patch.object(
                download_mit_bih_ecg,
                "download_wfdb_record",
                return_value=(),
            ) as download,
        ):
            destination = Path(directory)
            download_mit_bih_ecg.download_mit_bih_record(destination)

        self.assertEqual(
            (download_mit_bih_ecg.MIT_BIH_RECORD_100, destination),
            download.call_args.args,
        )


if __name__ == "__main__":
    unittest.main()
