import hashlib
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from sigvue.plugin import AnnotationRequest

from sigvue_examples.waterfall.analysis import _waterfall_spectrogram
from sigvue_examples.waterfall.workspace import create_workspace
from scripts.download_radio_astronomy import is_unpacked, md5, unpack
from scripts.generate_minimal_sigmf import write_sigmf


class WaterfallTests(unittest.TestCase):
    def test_waterfall_rows_cover_exact_full_fft_windows(self):
        waterfall, average, edges = _waterfall_spectrogram(
            np.ones(1_000, dtype=np.complex64),
            fft_size=10,
            maximum_rows=30,
        )
        self.assertEqual((30, 10), waterfall.shape)
        self.assertEqual((10,), average.shape)
        self.assertEqual((31,), edges.shape)
        self.assertEqual(0.0, edges[0])
        self.assertEqual(1_000.0, edges[-1])

    def test_waterfall_ignores_a_partial_tail_instead_of_zero_padding_it(self):
        sample_rate = 2_000_000.0
        samples = np.exp(1j * 2 * np.pi * 200_000 * np.arange(40_000) / sample_rate)
        waterfall, _, edges = _waterfall_spectrogram(
            samples.astype(np.complex64),
            fft_size=4_096,
            maximum_rows=200,
        )
        self.assertEqual((18, 4_096), waterfall.shape)
        self.assertEqual(38_912, edges[-1])
        self.assertLess(edges[-1], samples.size)
        np.testing.assert_allclose(
            10 ** (waterfall[0] / 10),
            10 ** (waterfall[-1] / 10),
            rtol=2e-5,
            atol=1e-14,
        )
        self.assertLess(float(np.min(waterfall)), -200.0)

    def test_windowed_waterfall_workspace_reads_sigmf_and_renders_spectrogram(self):
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

            workspace = create_workspace({"data_root": root})
            self.assertEqual(["survey"], [item.identifier for item in workspace.discover_items()])
            self.assertEqual("survey", workspace.discover_items()[0].title)
            opened = workspace.open_item("survey")
            self.assertEqual(
                "waterfall_annotation_region_color",
                opened.page.annotation.timeline_color_control,
            )
            self.assertEqual("windowed", opened.page.playback.mode)
            self.assertEqual("auto", opened.page.playback.time_unit)
            self.assertEqual("Received power (dBFS)", opened.page.playback.overview_label)
            self.assertEqual(400, len(opened.page.playback.overview_values))
            self.assertEqual((), opened.page.playback.overview_series)
            controls = {control.name: control for control in opened.page.controls}
            self.assertEqual("colormap", controls["waterfall_colormap"].control_type)
            self.assertEqual("limits", controls["waterfall_dbfs_limits"].control_type)
            self.assertEqual("Manual dBFS limits (Auto off)", controls["waterfall_dbfs_limits"].label)
            self.assertEqual((-90.0, -20.0), controls["waterfall_dbfs_limits"].default)
            self.assertEqual("Hann", controls["waterfall_fft_window"].default)
            self.assertEqual(50, controls["waterfall_overlap_percent"].default)
            self.assertEqual("toggle", controls["waterfall_auto_dbfs_scale"].control_type)
            self.assertTrue(controls["waterfall_auto_dbfs_scale"].default)
            self.assertEqual("toggle", controls["waterfall_show_annotations"].control_type)
            self.assertTrue(controls["waterfall_show_annotations"].default)
            self.assertEqual(1, controls["waterfall_slow_time_display_decimation"].default)
            self.assertEqual(1, controls["waterfall_fast_time_display_decimation"].default)
            self.assertEqual("Annotation display", controls["waterfall_annotation_region_color"].group)
            self.assertEqual("#ffffff", controls["waterfall_annotation_region_color"].default)
            self.assertEqual(0.5, controls["waterfall_annotation_region_width"].default)
            self.assertEqual(0.6, controls["waterfall_annotation_region_opacity"].default)
            self.assertEqual("solid", controls["waterfall_annotation_region_line_style"].default)

            figure = opened.page.views[0].callback({
                "waterfall_colormap": "Cividis",
                "waterfall_auto_dbfs_scale": "false",
                "waterfall_dbfs_limits": "-95,-15",
                "waterfall_fft_size": "1024",
                "waterfall_maximum_time_bins": "50",
            })
            self.assertEqual(
                ["scatter", "heatmap", "scatter", "scatter", "scatter"],
                [trace.type for trace in figure.data],
            )
            self.assertIn("Selection surface", [trace.name for trace in figure.data])
            self.assertEqual((-95.0, -15.0), tuple(figure.layout.yaxis.range))
            self.assertEqual((-95.0, -15.0), (figure.data[1].zmin, figure.data[1].zmax))
            self.assertEqual(figure.data[1].z.shape[0] + 1, len(figure.data[1].y))
            self.assertEqual(tuple(figure.layout.yaxis2.range), (figure.data[1].y[0], figure.data[1].y[-1]))
            self.assertEqual("#00224e", figure.data[1].colorscale[0][1])
            self.assertEqual("RF frequency (MHz)", figure.layout.xaxis2.title.text)
            self.assertEqual("Recording time (ms)", figure.layout.yaxis2.title.text)
            for axis in (figure.layout.xaxis, figure.layout.xaxis2, figure.layout.yaxis, figure.layout.yaxis2):
                self.assertIsNot(axis.fixedrange, True)
                self.assertIsNotNone(axis.uirevision)
                self.assertIsNone(axis.minallowed)
                self.assertIsNone(axis.maxallowed)
            self.assertEqual("bounded", opened.page.views[0].axis_navigation)
            region = figure.data[3]
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
            hit_targets = figure.data[4]
            self.assertIn("Imported IQEngine region", hit_targets.text[0])
            self.assertEqual("markers", hit_targets.mode)
            self.assertEqual(0.01, hit_targets.marker.opacity)

            decimated = opened.page.views[0].callback({
                "waterfall_fft_size": "1024",
                "waterfall_maximum_time_bins": "50",
                "waterfall_slow_time_display_decimation": "4",
                "waterfall_fast_time_display_decimation": "8",
            })
            np.testing.assert_array_equal(decimated.data[0].y, figure.data[0].y)
            np.testing.assert_array_equal(decimated.data[1].z, figure.data[1].z[::4, ::8])

            hidden = opened.page.views[0].callback({
                "waterfall_show_annotations": "false",
                "waterfall_fft_size": "1024",
                "waterfall_maximum_time_bins": "50",
            })
            self.assertEqual(
                ["scatter", "heatmap", "scatter"],
                [trace.type for trace in hidden.data],
            )
            self.assertFalse(hidden.layout.shapes)
            self.assertEqual(figure.layout.yaxis2.range, hidden.layout.yaxis2.range)
            self.assertNotEqual(figure.layout.uirevision, hidden.layout.uirevision)

            restyled = opened.page.views[0].callback({
                "waterfall_show_annotations": "false",
                "waterfall_colormap": "Inferno",
                "waterfall_fft_size": "1024",
                "waterfall_maximum_time_bins": "50",
            })
            self.assertEqual(hidden.layout.yaxis.range, restyled.layout.yaxis.range)
            self.assertNotEqual(hidden.layout.yaxis.uirevision, restyled.layout.yaxis.uirevision)

            moved = opened.page.views[0].callback({
                "__window_start_seconds": "0.1",
                "__window_end_seconds": "0.2",
                "waterfall_fft_size": "1024",
                "waterfall_maximum_time_bins": "50",
            })
            self.assertNotEqual(figure.layout.yaxis2.uirevision, moved.layout.yaxis2.uirevision)
            self.assertEqual((100.0, 197.28), tuple(moved.layout.yaxis2.range))

    def test_lte_workspace_uses_the_shared_sigmf_annotation_regions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            samples = np.exp(1j * 2 * np.pi * 8_000 * np.arange(2_000) / 100_000).astype(np.complex64)
            write_sigmf(root, "lte", samples, 100_000.0, "Annotated LTE fixture")
            metadata_path = root / "lte.sigmf-meta"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["captures"] = [{"core:sample_start": 0, "core:frequency": 806_000_000.0}]
            metadata.pop("annotations", None)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            opened = create_workspace({"data_root": root}).open_item("lte")
            self.assertIsNotNone(opened.page.annotation)
            self.assertEqual((), tuple(opened.page.annotation.discover_callback()))
            self.assertEqual(
                "waterfall_annotation_region_color",
                opened.page.annotation.timeline_color_control,
            )
            created = opened.page.annotation.annotate_callback({}, AnnotationRequest(
                0.0,
                values={
                    "start_seconds": "0.001",
                    "stop_seconds": "0.003",
                    "frequency_lower_hz": "805980000",
                    "frequency_upper_hz": "806020000",
                    "comment": "LTE allocation",
                },
            ))
            self.assertEqual("LTE allocation", created.comment)
            self.assertEqual((created,), tuple(opened.page.annotation.discover_callback()))
            figure = opened.page.views[0].callback({})
            self.assertEqual(
                ["scatter", "heatmap", "scatter", "scatter", "scatter"],
                [trace.type for trace in figure.data],
            )
            self.assertIn("Selection surface", [trace.name for trace in figure.data])
            self.assertEqual(figure.data[1].z.shape[0] + 1, figure.data[1].y.size)
            self.assertEqual(tuple(figure.layout.yaxis2.range), (figure.data[1].y[0], figure.data[1].y[-1]))
            self.assertIn("LTE allocation", figure.data[-1].text[0])
            fields = {field.name: field for field in opened.page.annotation.fields}
            self.assertEqual("waterfall-spectrum", fields["start_seconds"].plot_binding.view)
            self.assertEqual("yaxis2", fields["start_seconds"].plot_binding.axis)
            self.assertEqual("xaxis2", fields["frequency_lower_hz"].plot_binding.axis)

    def test_collection_waterfall_annotates_the_selected_member(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            samples = np.exp(1j * 2 * np.pi * 5_000 * np.arange(2_000) / 100_000).astype(np.complex64)
            write_sigmf(root, "first", samples, 100_000.0, "First member")
            write_sigmf(root, "second", samples, 100_000.0, "Second member")
            collection_path = root / "members.sigmf-collection"
            collection_path.write_text(json.dumps({
                "collection": {"name": "Two recordings"},
                "members": [
                    {"role": "calibration", "channel": 1, "metadata": "first.sigmf-meta"},
                    {"role": "ota", "channel": 1, "metadata": "second.sigmf-meta"},
                ],
            }), encoding="utf-8")

            workspace = create_workspace({
                "data_root": root,
                "source_type": "collection",
                "filename": "*.sigmf-collection",
            })
            opened = workspace.open_item("members")
            self.assertIsNotNone(opened.page.annotation)
            fields = {field.name: field for field in opened.page.annotation.fields}
            self.assertEqual("waterfall-member", fields["start_seconds"].plot_binding.view)
            created = opened.page.annotation.annotate_callback(
                {"__view_selection_waterfall-member": 1},
                AnnotationRequest(
                    0.0,
                    values={
                        "start_seconds": "0.001",
                        "stop_seconds": "0.003",
                        "frequency_lower_hz": "-10000",
                        "frequency_upper_hz": "10000",
                        "comment": "Selected member only",
                    },
                    view_selections={"waterfall-member": 1},
                ),
            )
            self.assertEqual("Selected member only", created.comment)
            first = json.loads((root / "first.sigmf-meta").read_text(encoding="utf-8"))
            second = json.loads((root / "second.sigmf-meta").read_text(encoding="utf-8"))
            self.assertEqual([], first["annotations"])
            self.assertEqual(1, len(second["annotations"]))
            discovered = tuple(opened.page.annotation.discover_callback())
            self.assertEqual(1, len(discovered))
            self.assertIn("OTA · Channel 1", discovered[0].label)
            self.assertEqual({"waterfall-member": 1}, discovered[0].view_selections)

    def test_collection_members_keep_their_own_duration_and_window_bounds(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_sigmf(root, "short", np.ones(100, dtype=np.complex64), 1_000.0, "Short")
            write_sigmf(root, "long", np.ones(1_000, dtype=np.complex64), 1_000.0, "Long")
            (root / "members.sigmf-collection").write_text(json.dumps({
                "collection": {"name": "Different lengths"},
                "members": [
                    {"role": "calibration", "channel": 1, "metadata": "short.sigmf-meta"},
                    {"role": "ota", "channel": 1, "metadata": "long.sigmf-meta"},
                ],
            }), encoding="utf-8")

            workspace = create_workspace({
                "data_root": root,
                "source_type": "collection",
                "filename": "*.sigmf-collection",
            })
            values = {
                "__window_start_seconds": "0.095",
                "__window_end_seconds": "0.12",
                "waterfall_fft_size": "1024",
            }
            opened = workspace.open_item_with_values("members", values)
            self.assertEqual((0.1, 1.0), opened.page.playback.overview_durations_seconds)
            short_figure, long_figure = (view.callback(values) for view in opened.page.views)
            self.assertEqual((75.0, 100.0), tuple(short_figure.layout.yaxis2.range))
            self.assertEqual((95.0, 120.0), tuple(long_figure.layout.yaxis2.range))
            self.assertEqual(2, len(short_figure.data[1].y))
            self.assertEqual(2, len(long_figure.data[1].y))

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
