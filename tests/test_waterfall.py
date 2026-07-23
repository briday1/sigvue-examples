from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go

from scripts.generate_minimal_sigmf import qam16, qpsk, write_sigmf
from scripts.generate_test_lte import generate as generate_lte
from sigvue_examples.comms.workspace import create_workspace as create_comms_workspace
from sigvue_examples.style import heatmap_grid_color
from sigvue_examples.waterfall.workspace import create_workspace as create_waterfall_workspace
from sigvue.web.application import SigvueApp


class CopyablePipelineTests(unittest.TestCase):
    def test_windowed_communications_pipeline_matches_bundled_example(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "examples:symbol_rate": 10_000.0,
                "examples:carrier_hz": 7_000.0,
                "examples:constellation_limit": 0.8,
                "examples:eye_limit": 0.9,
            }
            write_sigmf(root, "qpsk", qpsk(), 100_000.0, "Synthetic QPSK", **{
                **metadata, "examples:modulation": "QPSK",
            })
            write_sigmf(root, "16qam", qam16(), 100_000.0, "Synthetic 16-QAM", **{
                **metadata, "examples:modulation": "16-QAM",
            })

            workspace = create_comms_workspace({"data_root": root, "filename": "*.sigmf-meta"})
            items = workspace.discover_items()
            self.assertEqual({"Synthetic QPSK", "Synthetic 16-QAM"}, {item.title for item in items})
            for item in items:
                opened = workspace.open_item(item.identifier)
                self.assertEqual("windowed", opened.page.playback.mode)
                self.assertEqual("Mean received power (dBFS)", opened.page.playback.overview_label)
                self.assertEqual(["constellation", "eye"], [view.name for view in opened.page.views])
                constellation, eye = [view.callback({}) for view in opened.page.views]
                self.assertEqual("scattergl", constellation.data[0].type)
                self.assertEqual(["scattergl", "scattergl"], [trace.type for trace in eye.data])
                self.assertEqual((-0.8, 0.8), tuple(constellation.layout.xaxis.range))
                if "QPSK" in item.title:
                    symbols = np.asarray(constellation.data[0].x) + 1j * np.asarray(constellation.data[0].y)
                    unit = symbols / np.maximum(np.abs(symbols), 1e-12)
                    self.assertGreater(abs(np.mean(unit ** 4)), 0.8)

    def test_windowed_lte_pipeline_uses_rasterized_heatmap_and_quiet_grid(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            generate_lte(root)
            workspace = create_waterfall_workspace({"data_root": root})
            items = workspace.discover_items()
            self.assertEqual(2, len(items))

            opened = workspace.open_item(items[0].identifier)
            self.assertEqual("windowed", opened.page.playback.mode)
            self.assertIsNotNone(opened.page.annotation)
            self.assertEqual((), opened.page.annotation.discover_callback())
            controls = {control.name: control for control in opened.page.controls}
            self.assertEqual("select", controls["fft_size"].control_type)
            self.assertEqual("colormap", controls["colormap"].control_type)
            self.assertEqual("Raster rendering", controls["render_width"].group)
            self.assertEqual("mean", controls["render_aggregation"].default)
            self.assertEqual("Annotations", controls["show_annotations"].group)
            self.assertEqual("Annotations", controls["annotation_region_color"].group)
            self.assertEqual("Annotations", controls["annotation_region_width"].group)
            self.assertEqual("Annotations", controls["annotation_region_opacity"].group)

            figure = opened.page.views[0].callback({})
            self.assertEqual(["scatter", "heatmap"], [trace.type for trace in figure.data])
            self.assertEqual((2, 2), np.asarray(figure.data[1].z).shape)
            self.assertEqual(1, len(figure.layout.images))
            self.assertIsNotNone(figure.layout.xaxis2.range)
            self.assertEqual(
                (figure.layout.images[0].x, figure.layout.images[0].x + figure.layout.images[0].sizex),
                tuple(figure.layout.xaxis2.range),
            )
            self.assertEqual(heatmap_grid_color("light"), figure.layout.xaxis2.gridcolor)
            self.assertEqual(heatmap_grid_color("light"), figure.layout.yaxis2.gridcolor)
            self.assertEqual(0.35, figure.layout.xaxis2.gridwidth)

            metadata_path = next(root.glob("**/*.sigmf-meta"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["annotations"] = [{
                "core:sample_start": 10_000,
                "core:sample_count": 20_000,
                "core:comment": "Existing review region",
                "core:freq_lower_edge": 803_000_000.0,
                "core:freq_upper_edge": 805_000_000.0,
            }]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            annotated = workspace.open_item(metadata_path.stem.removesuffix(".sigmf"))
            self.assertEqual(1, len(annotated.page.annotation.discover_callback()))
            annotated_figure = annotated.page.views[0].callback({})
            annotation_traces = [trace for trace in annotated_figure.data if trace.name == "Annotations"]
            self.assertEqual(1, len(annotation_traces))
            self.assertEqual("scattergl", annotation_traces[0].type)
            self.assertEqual(
                [803.0, 805.0, 805.0, 803.0, 803.0, None],
                list(annotation_traces[0].x),
            )
            expected_start_ms = 10_000 / 30_720_000 * 1e3
            expected_stop_ms = 30_000 / 30_720_000 * 1e3
            np.testing.assert_allclose(
                list(annotation_traces[0].y)[:5],
                [expected_start_ms, expected_start_ms, expected_stop_ms, expected_stop_ms, expected_start_ms],
            )
            hover_trace = next(trace for trace in annotated_figure.data if trace.name == "Annotation details")
            self.assertEqual("%{text}<extra></extra>", hover_trace.hovertemplate)
            self.assertIn("Existing review region", hover_trace.text[0])
            self.assertIn("803–805 MHz", hover_trace.text[0])
            self.assertAlmostEqual((expected_start_ms + expected_stop_ms) / 2, hover_trace.y[0])
            customized = annotated.page.views[0].callback({
                "annotation_region_color": "#ff8800",
                "annotation_region_width": "3.5",
                "annotation_region_opacity": "0.35",
            })
            customized_trace = next(trace for trace in customized.data if trace.name == "Annotations")
            self.assertEqual("#ff8800", customized_trace.line.color)
            self.assertEqual(3.5, customized_trace.line.width)
            self.assertEqual(0.35, customized_trace.opacity)
            hidden = annotated.page.views[0].callback({"show_annotations": "false"})
            self.assertFalse(any(trace.name in {"Annotations", "Annotation details"} for trace in hidden.data))

    def test_empty_comms_recording_still_exposes_annotation_capability(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_sigmf(root, "qpsk", qpsk(), 100_000.0, "Synthetic QPSK", **{
                "examples:symbol_rate": 10_000.0,
                "examples:modulation": "QPSK",
            })
            workspace = create_comms_workspace({"data_root": root, "filename": "*.sigmf-meta"})
            opened = workspace.open_item(workspace.discover_items()[0].identifier)
            self.assertIsNotNone(opened.page.annotation)
            self.assertEqual((), opened.page.annotation.discover_callback())

    def test_lazy_comms_workspace_only_builds_the_selected_tab(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_sigmf(
                root,
                "qpsk",
                qpsk(),
                100_000.0,
                "Synthetic QPSK",
                **{
                    "examples:symbol_rate": 10_000.0,
                    "examples:modulation": "QPSK",
                },
            )
            workspace = create_comms_workspace({
                "data_root": root,
                "filename": "*.sigmf-meta",
            })
            item_id = workspace.discover_items()[0].identifier
            app = SigvueApp()
            app.register_workspace(workspace)

            with (
                patch(
                    "sigvue_examples.comms.presentation.constellation_figure",
                    return_value=go.Figure(),
                ) as constellation,
                patch(
                    "sigvue_examples.comms.presentation.eye_figure",
                    return_value=go.Figure(),
                ) as eye,
            ):
                initial = app.open_item(workspace.metadata.identifier, item_id)
                switched = app.open_item(
                    workspace.metadata.identifier,
                    item_id,
                    {"__view_selection___tabs": "1"},
                )

            self.assertTrue(workspace.lazy_views)
            self.assertEqual(
                ["constellation"],
                [view["name"] for view in initial["page"]["rendered_views"]],
            )
            self.assertEqual(
                ["eye"],
                [view["name"] for view in switched["page"]["rendered_views"]],
            )
            constellation.assert_called_once()
            eye.assert_called_once()


if __name__ == "__main__":
    unittest.main()
