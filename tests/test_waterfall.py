import hashlib
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scientific_workspace_examples.waterfall import create_radio_astronomy_workspace
from scripts.download_radio_astronomy import is_unpacked, md5, unpack
from scripts.generate_minimal_sigmf import write_sigmf


class WaterfallTests(unittest.TestCase):
    def test_windowed_rfi_workspace_reads_sigmf_and_renders_spectrogram(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(8242048)
            samples = np.asarray(
                0.04 * (rng.normal(size=50_000) + 1j * rng.normal(size=50_000))
                + 0.35 * np.exp(1j * 2 * np.pi * 12_000 * np.arange(50_000) / 100_000),
                dtype=np.complex64,
            )
            write_sigmf(root, "survey", samples, 100_000.0, "Synthetic ATA RFI test fixture")
            metadata_path = root / "survey.sigmf-meta"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["global"]["core:description"] = ""
            metadata["captures"] = [{"core:sample_start": 0, "core:frequency": 1_420_000_000.0}]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            workspace = create_radio_astronomy_workspace({"data_root": root})
            self.assertEqual(["survey"], [item.identifier for item in workspace.discover_items()])
            self.assertEqual("survey", workspace.discover_items()[0].title)
            opened = workspace.open_item("survey")
            self.assertEqual("windowed", opened.page.playback.mode)
            self.assertEqual("Sampled wideband power (dBFS)", opened.page.playback.overview_label)
            self.assertEqual(400, len(opened.page.playback.overview_values))
            controls = {control.name: control for control in opened.page.controls}
            self.assertEqual("colormap", controls["rfi_colormap"].control_type)
            self.assertEqual("limits", controls["rfi_dbfs_limits"].control_type)

            figure = opened.page.views[0].callback({
                "rfi_colormap": "Cividis",
                "rfi_dbfs_limits": "-95,-15",
                "rfi_fft_size": "1024",
                "rfi_maximum_time_bins": "50",
            })
            self.assertEqual(["scatter", "heatmap"], [trace.type for trace in figure.data])
            self.assertEqual((-95.0, -15.0), (figure.data[1].zmin, figure.data[1].zmax))
            self.assertEqual("#00224e", figure.data[1].colorscale[0][1])
            self.assertEqual("RF frequency (MHz)", figure.layout.xaxis2.title.text)

    def test_download_helpers_verify_and_safely_unpack_tar_archive(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "fixture.sigmf"
            payload = b'{"global": {"core:datatype": "ci16_le"}}'
            with tarfile.open(archive, "w") as bundle:
                info = tarfile.TarInfo("fixture/fixture.sigmf-meta")
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
            self.assertEqual(hashlib.md5(archive.read_bytes()).hexdigest(), md5(archive))
            output = root / "unpacked"
            output.mkdir()
            unpack(archive, output)
            self.assertEqual(payload, (output / "fixture/fixture.sigmf-meta").read_bytes())
            (output / "fixture/fixture.sigmf-data").write_bytes(b"data")
            self.assertTrue(is_unpacked({"key": "fixture.sigmf"}, output))


if __name__ == "__main__":
    unittest.main()
