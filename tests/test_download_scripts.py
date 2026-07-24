import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sigvue.helpers import RemoteFile

from scripts import download_lte_sigmf, download_radio_astronomy


class DownloadScriptTests(unittest.TestCase):
    def test_get_all_data_includes_ecg_and_weather_downloaders(self):
        repository = Path(__file__).resolve().parents[1]
        aggregate = (
            repository / "scripts/get_all_data.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("download_mit_bih_ecg.py", aggregate)
        self.assertIn("download_weather_radar.py", aggregate)

    def test_lte_manifest_downloads_each_verified_file_to_its_workspace(self):
        with TemporaryDirectory() as directory, patch.object(
            download_lte_sigmf,
            "download_file",
        ) as download:
            root = Path(directory)
            download.side_effect = (
                lambda remote, destination, **kwargs:
                Path(destination) / remote.filename
            )
            paths = download_lte_sigmf.download_lte_recordings(root)

        self.assertEqual(4, len(paths))
        self.assertTrue(
            all(
                isinstance(remote, RemoteFile)
                and remote.size is not None
                and remote.checksum is not None
                for files in download_lte_sigmf.LTE_MANIFEST.values()
                for remote in files
            )
        )
        self.assertEqual(
            {
                root / "lte/downlink",
                root / "lte/uplink",
            },
            {call.args[1] for call in download.call_args_list},
        )
        for call in download.call_args_list:
            self.assertEqual(
                call.args[0].filename.endswith(".sigmf-meta"),
                call.kwargs["preserve_existing"],
            )

    def test_radio_astronomy_uses_shared_verified_download_and_safe_extract(self):
        remote = RemoteFile(
            "https://example.test/capture.sigmf",
            "capture.sigmf",
            size=4,
            checksum=f"md5:{'0' * 32}",
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / remote.filename
            archive.write_bytes(b"test")
            with (
                patch.object(
                    download_radio_astronomy,
                    "record_files",
                    return_value=[remote],
                ),
                patch.object(
                    download_radio_astronomy,
                    "download_file",
                    return_value=archive,
                ) as download,
                patch.object(
                    download_radio_astronomy,
                    "safe_extract_tar",
                ) as extract,
                patch(
                    "sys.argv",
                    [
                        "download_radio_astronomy.py",
                        "--output",
                        str(root),
                    ],
                ),
            ):
                download_radio_astronomy.main()

        download.assert_called_once()
        extract.assert_called_once_with(archive, root)
        self.assertFalse(archive.exists())

    def test_get_all_data_fails_early_with_an_editable_install_command(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            fake_bin = Path(directory)
            fake_python = fake_bin / "python"
            fake_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            fake_python.chmod(0o755)
            result = subprocess.run(
                ["bash", str(repository / "scripts/get_all_data.sh")],
                capture_output=True,
                check=False,
                env={
                    **os.environ,
                    "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                },
                text=True,
            )

        self.assertEqual(1, result.returncode)
        self.assertIn(
            f'python -m pip install -e "{repository}"',
            result.stderr,
        )
        self.assertNotIn("==>", result.stdout)


if __name__ == "__main__":
    unittest.main()
