import hashlib
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from sigvue_examples.waterfall import _rfi_spectrogram, create_radio_astronomy_workspace
from scripts.download_radio_astronomy import is_unpacked, md5, unpack
from scripts.generate_minimal_sigmf import write_sigmf


class WaterfallTests(unittest.TestCase):
    def test_rfi_display_rows_cover_the_entire_selected_buffer(self):
        waterfall, average, centers = _rfi_spectrogram(
            np.ones(1_000, dtype=np.complex64),
            fft_size=10,
            maximum_rows=30,
        )
        self.assertEqual((30, 10), waterfall.shape)
        self.assertEqual((10,), average.shape)
        self.assertEqual(5.0, centers[0])
        self.assertEqual(995.0, centers[-1])

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
            metadata["annotations"] = [{
                "core:sample_start": 1_000,
                "core:sample_count": 500,
                "core:freq_lower_edge": 1_419_990_000.0,
                "core:freq_upper_edge": 1_420_020_000.0,
                "core:comment": "Imported IQEngine region",
            }]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            workspace = create_radio_astronomy_workspace({"data_root": root})
            self.assertEqual(["survey"], [item.identifier for item in workspace.discover_items()])
            self.assertEqual("survey", workspace.discover_items()[0].title)
            opened = workspace.open_item("survey")
            self.assertEqual(
                "rfi_annotation_region_color",
                opened.page.annotation.timeline_color_control,
            )
            self.assertEqual("windowed", opened.page.playback.mode)
            self.assertEqual("auto", opened.page.playback.time_unit)
            self.assertEqual("Sampled wideband power (dBFS)", opened.page.playback.overview_label)
            self.assertEqual(400, len(opened.page.playback.overview_values))
            controls = {control.name: control for control in opened.page.controls}
            self.assertEqual("colormap", controls["rfi_colormap"].control_type)
            self.assertEqual("limits", controls["rfi_dbfs_limits"].control_type)
            self.assertEqual("toggle", controls["rfi_show_annotations"].control_type)
            self.assertTrue(controls["rfi_show_annotations"].default)
            self.assertEqual("Annotation display", controls["rfi_annotation_region_color"].group)
            self.assertEqual("#ffffff", controls["rfi_annotation_region_color"].default)
            self.assertEqual(0.5, controls["rfi_annotation_region_width"].default)
            self.assertEqual(0.6, controls["rfi_annotation_region_opacity"].default)
            self.assertEqual("solid", controls["rfi_annotation_region_line_style"].default)

            figure = opened.page.views[0].callback({
                "rfi_colormap": "Cividis",
                "rfi_dbfs_limits": "-95,-15",
                "rfi_fft_size": "1024",
                "rfi_maximum_time_bins": "50",
            })
            self.assertEqual(["scatter", "heatmap", "scatter", "scatter"], [trace.type for trace in figure.data])
            self.assertEqual((-95.0, -15.0), (figure.data[1].zmin, figure.data[1].zmax))
            self.assertEqual("#00224e", figure.data[1].colorscale[0][1])
            self.assertEqual("RF frequency (MHz)", figure.layout.xaxis2.title.text)
            self.assertEqual("Recording time (ms)", figure.layout.yaxis2.title.text)
            for axis in (figure.layout.xaxis, figure.layout.xaxis2, figure.layout.yaxis, figure.layout.yaxis2):
                self.assertIsNot(axis.fixedrange, True)
                self.assertEqual(float(axis.range[0]), float(axis.minallowed))
                self.assertEqual(float(axis.range[1]), float(axis.maxallowed))
                self.assertEqual(float(axis.range[0]), float(axis.autorangeoptions.clipmin))
                self.assertEqual(float(axis.range[1]), float(axis.autorangeoptions.clipmax))
            region = figure.data[2]
            self.assertFalse(region.showlegend)
            self.assertEqual("skip", region.hoverinfo)
            self.assertEqual("rgba(255,255,255,0.6)", region.line.color)
            self.assertEqual(0.5, region.line.width)
            self.assertEqual("solid", region.line.dash)
            self.assertFalse(figure.layout.shapes)
            y_values = np.asarray([value for value in region.y if value is not None])
            waterfall_range = tuple(float(value) for value in figure.layout.yaxis2.range)
            self.assertGreaterEqual(float(np.min(y_values)), waterfall_range[0])
            self.assertLessEqual(float(np.max(y_values)), waterfall_range[1])
            self.assertGreaterEqual(
                float(np.max(y_values) - np.min(y_values)),
                5.0,
            )
            hit_targets = figure.data[3]
            self.assertIn("Imported IQEngine region", hit_targets.text[0])
            self.assertEqual("markers", hit_targets.mode)
            self.assertEqual(0.01, hit_targets.marker.opacity)

            hidden = opened.page.views[0].callback({
                "rfi_show_annotations": "false",
                "rfi_fft_size": "1024",
                "rfi_maximum_time_bins": "50",
            })
            self.assertEqual(["scatter", "heatmap"], [trace.type for trace in hidden.data])
            self.assertFalse(hidden.layout.shapes)
            self.assertEqual(figure.layout.yaxis2.range, hidden.layout.yaxis2.range)
            self.assertNotEqual(figure.layout.uirevision, hidden.layout.uirevision)

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
