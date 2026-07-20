import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sigvue.web.application import create_app
from sigvue.profile import load_browser_profile

import sigvue_examples.io.sigmf.recording as sigmf
import sigvue_examples.style as style
from sigvue_examples.comms.workspace import create_workspace as create_comms_workspace
from sigvue_examples.io.sigmf.recording import load_recording
from sigvue_examples.waterfall.workspace import create_workspace as create_waterfall_workspace
from scripts.generate_segmented_results import EVENTS, generate as generate_segmented_results
from scripts.generate_minimal_sigmf import qam16, qpsk, write_sigmf


ROOT = Path(__file__).resolve().parents[1]


class MinimalExampleTests(unittest.TestCase):
    def test_every_workspace_declares_standard_discovery_columns(self):
        app = create_app(config_path=ROOT / "browser.toml")
        for workspace in app.list_workspaces():
            listing = app.browse_items(workspace["id"], {})
            self.assertEqual(
                ["date", "sample_rate", "rf_frequency"],
                [column["key"] for column in listing["columns"]],
                workspace["id"],
            )

    def test_shared_modules_do_not_own_browser_contracts(self):
        self.assertNotIn("sigvue", inspect.getsource(sigmf))
        self.assertNotIn("sigvue", inspect.getsource(style))
        self.assertFalse(hasattr(sigmf, "SigMFWindow"))
        self.assertFalse(hasattr(sigmf, "WindowedSigMF"))
        self.assertFalse(hasattr(sigmf, "WholeSigMF"))

    def test_shared_plot_styles_enable_subtle_light_grid(self):
        import plotly.graph_objects as go

        for figure in (
            style.style_figure(go.Figure(), "light", "Example"),
            style.style_plotly(go.Figure(), theme="light"),
        ):
            self.assertTrue(figure.layout.xaxis.showgrid)
            self.assertTrue(figure.layout.yaxis.showgrid)
            self.assertEqual(style.GRID, figure.layout.xaxis.gridcolor)
            self.assertEqual(style.GRID, figure.layout.yaxis.gridcolor)
            self.assertEqual(0.5, figure.layout.xaxis.gridwidth)
            self.assertEqual(0.5, figure.layout.yaxis.gridwidth)

    def test_profile_loads_example_workspaces(self):
        app = create_app(config_path=ROOT / "browser.toml")
        self.assertEqual(
            [
                "digital-comms",
                "downloaded-waterfall",
                "acoustic-events-segmented",
                "radio-astronomy-rfi",
                "lte-recordings",
                "lfm-live",
                "radar-waterfall",
            ],
            [workspace["id"] for workspace in app.list_workspaces()],
        )
        expected_mode_tags = {
            "digital-comms": "windowed",
            "downloaded-waterfall": "windowed",
            "acoustic-events-segmented": "segmented",
            "radio-astronomy-rfi": "windowed",
            "lte-recordings": "windowed",
            "lfm-live": "live",
            "radar-waterfall": "windowed",
        }
        for workspace in app.list_workspaces():
            self.assertIn(expected_mode_tags[workspace["id"]], workspace["tags"])
            self.assertIn(expected_mode_tags[workspace["id"]], workspace["description"].lower())

    def test_same_waterfall_factory_creates_distinct_configured_instances(self):
        profile = load_browser_profile(ROOT / "browser.toml")
        waterfall_specs = [
            spec
            for spec in profile.workspaces
            if spec.attribute == "create_workspace" and spec.module_name.endswith(".waterfall.workspace")
        ]
        self.assertEqual(4, len(waterfall_specs))
        self.assertEqual(1, len({(spec.module_name, spec.attribute) for spec in waterfall_specs}))
        self.assertEqual(
            {"downloaded-waterfall", "radio-astronomy-rfi", "lte-recordings", "radar-waterfall"},
            {spec.metadata_overrides["identifier"] for spec in waterfall_specs},
        )

        app = create_app(config_path=ROOT / "browser.toml")
        workspaces = {workspace["id"]: workspace for workspace in app.list_workspaces()}
        self.assertEqual("Downloaded", workspaces["downloaded-waterfall"]["name"])
        self.assertIn("downloaded", workspaces["downloaded-waterfall"]["tags"])
        self.assertEqual("radio astronomy", workspaces["radio-astronomy-rfi"]["category"])

    def test_communications_recordings_are_file_backed_and_windowed(self):
        app = create_app(config_path=ROOT / "browser.toml")
        items = app.list_items("digital-comms", {})
        self.assertTrue({"16qam", "qpsk"}.issubset({item["id"] for item in items}))
        for item in items:
            self.assertTrue(Path(item["source_reference"]).is_file())
            page = app.open_item("digital-comms", item["id"])["page"]
            self.assertEqual("windowed", page["playback"]["mode"])
            expected_unit = "samples" if "Sample-normalized" in item["subtitle"] else "ms"
            self.assertEqual(expected_unit, page["playback"]["time_unit"])
            self.assertGreaterEqual(len(page["rendered_views"]), 1)

    def test_null_sample_rate_uses_normalized_sample_coordinates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            samples = qpsk(sample_rate=10_000.0, duration=0.1)
            write_sigmf(root, "normalized-qpsk", samples, 10_000.0, "QPSK without a known rate")
            metadata_path = root / "normalized-qpsk.sigmf-meta"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["global"]["core:sample_rate"] = None
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            workspace = create_comms_workspace({"data_root": root})
            item = workspace.discover_items()[0]
            self.assertEqual("Sample-normalized (rate unavailable)", item.subtitle)
            opened = workspace.open_item(item.identifier)
            self.assertEqual("samples", opened.page.playback.time_unit)
            self.assertEqual(float(samples.size), opened.page.playback.duration_seconds)
            self.assertEqual("Normalized samples", opened.page.statistics["Coordinate basis"])

    def test_qpsk_window_has_received_power_overview(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = next(item for item in app.list_items("digital-comms", {}) if item["id"] == "qpsk")
        values = {"__window_start_seconds": "0.03", "__window_end_seconds": "0.04"}
        opened = app.open_item("digital-comms", item["id"], values)
        playback = opened["page"]["playback"]
        self.assertEqual("Received power", playback["overview_label"])
        self.assertEqual(200, len(playback["overview_values"]))

    def test_qpsk_shows_constellation_then_eye(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = next(item for item in app.list_items("digital-comms", {}) if item["id"] == "qpsk")
        page = app.open_item("digital-comms", item["id"])["page"]
        self.assertEqual(
            ["constellation", "eye"],
            [view["name"] for view in page["rendered_views"]],
        )
        constellation = page["rendered_views"][0]["value"]
        self.assertEqual([-0.8, 0.8], constellation["layout"]["xaxis"]["range"])
        self.assertEqual([-0.8, 0.8], constellation["layout"]["yaxis"]["range"])
        eye = page["rendered_views"][1]["value"]
        self.assertEqual([0, 2], eye["layout"]["xaxis"]["range"])
        self.assertEqual([-0.9, 0.9], eye["layout"]["yaxis"]["range"])

    def test_16qam_uses_same_constellation_and_eye_views(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = next(item for item in app.list_items("digital-comms", {}) if item["id"] == "16qam")
        page = app.open_item("digital-comms", item["id"])["page"]
        self.assertEqual(["constellation", "eye"], [view["name"] for view in page["rendered_views"]])
        self.assertEqual("16-QAM", page["statistics"]["Modulation"])

    def test_lte_recording_renders_rf_spectrum_and_waterfall(self):
        app = create_app(config_path=ROOT / "browser.toml")
        items = app.list_items("lte-recordings", {})
        self.assertEqual(2, len(items))
        item = next(item for item in items if "downlink" in item["id"])
        page = app.open_item("lte-recordings", item["id"])["page"]
        self.assertEqual("windowed", page["playback"]["mode"])
        self.assertEqual("auto", page["playback"]["time_unit"])
        self.assertEqual("Received power (dBFS)", page["playback"]["overview_label"])
        self.assertEqual(400, len(page["playback"]["overview_values"]))
        self.assertEqual(["waterfall-spectrum"], [view["name"] for view in page["rendered_views"]])
        figure = page["rendered_views"][0]["value"]
        self.assertEqual(["scatter", "heatmap"], [trace["type"] for trace in figure["data"][:2]])
        self.assertEqual("RF frequency (MHz)", figure["layout"]["xaxis2"]["title"]["text"])
        self.assertEqual("Recording time (ms)", figure["layout"]["yaxis2"]["title"]["text"])
        self.assertEqual("#0d0887", figure["data"][1]["colorscale"][0][1])
        colormap = next(control for control in page["controls"] if control["name"] == "waterfall_colormap")
        self.assertEqual("colormap", colormap["control_type"])
        self.assertEqual(10, len(colormap["options"]))
        self.assertEqual(10, len(colormap["option_previews"]))
        self.assertEqual("Plasma", colormap["default"])
        self.assertEqual("details", colormap["placement"])
        controls = {control["name"]: control for control in page["controls"]}
        self.assertTrue(controls["waterfall_auto_dbfs_scale"]["default"])
        self.assertEqual((-90.0, -20.0), controls["waterfall_dbfs_limits"]["default"])
        self.assertEqual(4096, controls["waterfall_fft_size"]["default"])
        self.assertEqual("Hann", controls["waterfall_fft_window"]["default"])
        self.assertEqual(50, controls["waterfall_overlap_percent"]["default"])
        self.assertEqual(200, controls["waterfall_maximum_time_bins"]["default"])
        self.assertTrue(controls["waterfall_show_annotations"]["default"])
        self.assertEqual(0.6, controls["waterfall_annotation_region_opacity"]["default"])
        self.assertEqual("806 MHz", page["statistics"]["Center frequency"])

        changed = app.open_item("lte-recordings", item["id"], {"waterfall_colormap": "Cividis"})["page"]
        changed_scale = changed["rendered_views"][0]["value"]["data"][1]["colorscale"]
        self.assertEqual("#00224e", changed_scale[0][1])

        changed = app.open_item("lte-recordings", item["id"], {
            "waterfall_auto_dbfs_scale": "false",
            "waterfall_dbfs_limits": "-92,-22",
        })["page"]
        changed_figure = changed["rendered_views"][0]["value"]
        self.assertEqual([-92.0, -22.0], changed_figure["layout"]["yaxis"]["range"])
        self.assertEqual((-92.0, -22.0), (changed_figure["data"][1]["zmin"], changed_figure["data"][1]["zmax"]))

        recording = load_recording(
            ROOT / "data/lte/downlink/LTE_downlink_806MHz_2022-04-09_30720ksps.sigmf-meta"
        )
        self.assertEqual("ci16_le", recording.datatype)
        self.assertEqual((1, 16), recording.read(0, 16).shape)
        self.assertLessEqual(float(abs(recording.read(0, 16)).max()), 1.0)

    def test_lte_uplink_uses_its_own_recording(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = next(item for item in app.list_items("lte-recordings", {}) if "uplink" in item["id"])
        self.assertIn("LTE_uplink_847MHz", item["id"])
        page = app.open_item("lte-recordings", item["id"])["page"]
        self.assertEqual("windowed", page["playback"]["mode"])
        self.assertEqual("847 MHz", page["statistics"]["Center frequency"])

    def test_lfm_live_workspace_discovers_both_bandwidth_collections(self):
        app = create_app(config_path=ROOT / "browser.toml")
        items = app.list_items("lfm-live", {})
        self.assertEqual(
            {"lfm-10mhz", "lfm-2mhz"},
            {item["id"] for item in items},
        )
        sample_rates = {
            app.open_item("lfm-live", item["id"])["page"]["statistics"]["Sample rate"]
            for item in items
        }
        self.assertEqual({"10 MHz", "2 MHz"}, sample_rates)

    def test_generic_waterfall_reuses_radar_ota_sigmf_members(self):
        app = create_app(config_path=ROOT / "browser.toml")
        items = app.list_items("radar-waterfall", {})
        self.assertEqual(2, len(items))
        self.assertEqual({"lfm-2mhz", "lfm-10mhz"}, {item["id"] for item in items})
        self.assertTrue(all(item["subtitle"] == "12 collection members" for item in items))

        page = app.open_item("radar-waterfall", items[0]["id"])["page"]
        self.assertEqual("windowed", page["playback"]["mode"])
        self.assertTrue(page["annotation"]["enabled"])
        self.assertTrue(
            all(
                field["plot_binding"]["view"] == "waterfall-member"
                for field in page["annotation"]["fields"]
                if field["plot_binding"] is not None
            )
        )
        self.assertEqual("Received power (dBFS)", page["playback"]["overview_label"])
        self.assertEqual("waterfall-member", page["playback"]["overview_switcher_key"])
        self.assertEqual(12, len(page["playback"]["overview_series"]))
        self.assertTrue(all(len(series) == 400 for series in page["playback"]["overview_series"]))
        self.assertEqual(
            (0.1,) * 8 + (1.0,) * 4,
            page["playback"]["overview_durations_seconds"],
        )
        self.assertEqual(12, len(page["rendered_views"]))
        self.assertTrue(all(view["axis_navigation"] == "bounded" for view in page["rendered_views"]))
        self.assertEqual(
            [f"waterfall-member-{index}" for index in range(12)],
            [view["name"] for view in page["rendered_views"]],
        )
        switcher = page["layout"]["children"][0]
        self.assertEqual("view_switcher", switcher["kind"])
        self.assertEqual("dropdown", switcher["props"]["selector"])
        self.assertEqual(
            [
                *(f"Calibration · Channel {channel}" for channel in range(1, 5)),
                *(f"Terminated noise · Channel {channel}" for channel in range(1, 5)),
                *(f"OTA · Channel {channel}" for channel in range(1, 5)),
            ],
            [child["props"]["label"] for child in switcher["children"]],
        )
        figure = page["rendered_views"][0]["value"]
        self.assertEqual(["scatter", "heatmap"], [trace["type"] for trace in figure["data"][:2]])
        calibration_time_range = page["rendered_views"][0]["value"]["layout"]["yaxis2"]["range"]
        ota_time_range = page["rendered_views"][8]["value"]["layout"]["yaxis2"]["range"]
        self.assertAlmostEqual(19.8656, calibration_time_range[1] - calibration_time_range[0])
        self.assertAlmostEqual(19.8656, ota_time_range[1] - ota_time_range[0])
        calibration_dbfs_range = page["rendered_views"][0]["value"]["layout"]["yaxis"]["range"]
        noise_dbfs_range = page["rendered_views"][4]["value"]["layout"]["yaxis"]["range"]
        self.assertNotEqual(calibration_dbfs_range, noise_dbfs_range)

        wide = app.open_item(
            "radar-waterfall",
            items[0]["id"],
            {"__window_start_seconds": "0", "__window_end_seconds": "0.5"},
        )["page"]
        wide_calibration_range = wide["rendered_views"][0]["value"]["layout"]["yaxis2"]["range"]
        wide_ota_range = wide["rendered_views"][8]["value"]["layout"]["yaxis2"]["range"]
        self.assertAlmostEqual(99.9424, wide_calibration_range[1] - wide_calibration_range[0])
        self.assertAlmostEqual(499.9168, wide_ota_range[1] - wide_ota_range[0])

    def test_single_recording_waterfall_does_not_show_member_switcher(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_sigmf(root, "single", qpsk(duration=0.01), 100_000.0, "Single recording")
            workspace = create_waterfall_workspace({"data_root": root, "filename": "single.sigmf-meta"})
            page = workspace.open_item("single").page
            self.assertEqual(["waterfall-spectrum"], [view.name for view in page.views])
            self.assertEqual("grid", page.layout.kind)
            self.assertFalse(any(node.kind == "view_switcher" for node in page.layout.children))

    def test_segmented_acoustic_workspace_displays_irregular_stored_results(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = app.list_items("acoustic-events-segmented", {})[0]
        initial = app.open_item("acoustic-events-segmented", item["id"])["page"]
        self.assertEqual("segmented", initial["playback"]["mode"])
        starts = [segment["start_seconds"] for segment in initial["playback"]["segments"]]
        self.assertEqual([event[1] for event in EVENTS], starts)
        self.assertEqual(["waveform", "spectrum"], [view["name"] for view in initial["rendered_views"]])

        selected = app.open_item(
            "acoustic-events-segmented",
            item["id"],
            {"__segment_id": "event-005"},
        )["page"]
        self.assertEqual("event-005", selected["playback"]["selected_segment_id"])
        self.assertEqual("Valve actuation", selected["statistics"]["Stored event"])

    def test_sigmf_reader_loads_only_requested_frames(self):
        recording = load_recording(ROOT / "data/comms/qpsk.sigmf-meta")
        samples = recording.read(25, 100)
        self.assertEqual((1, 100), samples.shape)
        self.assertEqual((1, 10), recording.read(recording.sample_count - 10, 100).shape)

    def test_minimal_data_can_be_regenerated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_sigmf(root, "qpsk", qpsk(duration=0.01), 100_000.0, "QPSK")
            write_sigmf(root, "16qam", qam16(duration=0.01), 100_000.0, "16-QAM")
            self.assertEqual(1_000, load_recording(root / "qpsk.sigmf-meta").sample_count)
            self.assertEqual(1_000, load_recording(root / "16qam.sigmf-meta").sample_count)
            results = generate_segmented_results(root / "acoustic-events.json")
            self.assertTrue(results.is_file())


if __name__ == "__main__":
    unittest.main()
