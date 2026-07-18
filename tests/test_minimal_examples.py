import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workspace_browser.web.application import create_app

from scientific_workspace_examples.sigmf import load_recording
from scripts.generate_segmented_results import EVENTS, generate as generate_segmented_results
from scripts.generate_minimal_sigmf import multiple_tones, qpsk, write_sigmf


ROOT = Path(__file__).resolve().parents[1]


class MinimalExampleTests(unittest.TestCase):
    def test_profile_loads_three_compact_and_two_lfm_workspaces(self):
        app = create_app(config_path=ROOT / "browser.toml")
        self.assertEqual(
            ["qpsk-windowed", "acoustic-events-segmented", "multi-tone-seek", "lfm-live", "lfm-static"],
            [workspace["id"] for workspace in app.list_workspaces()],
        )
        expected_mode_tags = {
            "qpsk-windowed": "windowed",
            "acoustic-events-segmented": "segmented",
            "multi-tone-seek": "seek",
            "lfm-live": "live",
            "lfm-static": "static",
        }
        for workspace in app.list_workspaces():
            self.assertIn(expected_mode_tags[workspace["id"]], workspace["tags"])
            self.assertIn(expected_mode_tags[workspace["id"]], workspace["description"].lower())

    def test_compact_recordings_are_file_backed_and_use_distinct_modes(self):
        app = create_app(config_path=ROOT / "browser.toml")
        expected = {"qpsk-windowed": "windowed", "multi-tone-seek": "seek"}
        for workspace_id, expected_mode in expected.items():
            items = app.list_items(workspace_id, {})
            self.assertEqual(1, len(items))
            self.assertTrue(Path(items[0]["source_reference"]).is_file())
            page = app.open_item(workspace_id, items[0]["id"])["page"]
            self.assertEqual(expected_mode, page["playback"]["mode"])
            self.assertGreaterEqual(len(page["rendered_views"]), 1)

    def test_qpsk_window_has_received_power_overview(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = app.list_items("qpsk-windowed", {})[0]
        values = {"__window_start_seconds": "0.03", "__window_end_seconds": "0.04"}
        opened = app.open_item("qpsk-windowed", item["id"], values)
        playback = opened["page"]["playback"]
        self.assertEqual("Received power", playback["overview_label"])
        self.assertEqual(200, len(playback["overview_values"]))

    def test_qpsk_shows_constellation_then_eye(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = app.list_items("qpsk-windowed", {})[0]
        page = app.open_item("qpsk-windowed", item["id"])["page"]
        self.assertEqual(
            ["constellation", "eye"],
            [view["name"] for view in page["rendered_views"]],
        )
        constellation = page["rendered_views"][0]["value"]
        self.assertEqual([-0.75, 0.75], constellation["layout"]["xaxis"]["range"])
        self.assertEqual([-0.75, 0.75], constellation["layout"]["yaxis"]["range"])
        eye = page["rendered_views"][1]["value"]
        self.assertEqual([0, 2], eye["layout"]["xaxis"]["range"])
        self.assertEqual([-1, 1], eye["layout"]["yaxis"]["range"])

    def test_multi_tone_uses_one_plotly_figure_for_psd_and_waterfall(self):
        app = create_app(config_path=ROOT / "browser.toml")
        item = app.list_items("multi-tone-seek", {})[0]
        page = app.open_item("multi-tone-seek", item["id"])["page"]
        self.assertEqual(["tones"], [view["name"] for view in page["rendered_views"]])
        figure = page["rendered_views"][0]["value"]
        self.assertEqual(["scatter", "heatmap"], [trace["type"] for trace in figure["data"]])

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
        recording = load_recording(ROOT / "data/qpsk.sigmf-meta")
        samples = recording.read(25, 100)
        self.assertEqual((1, 100), samples.shape)
        self.assertEqual((1, 10), recording.read(recording.sample_count - 10, 100).shape)

    def test_minimal_data_can_be_regenerated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_sigmf(root, "qpsk", qpsk(duration=0.01), 100_000.0, "QPSK")
            write_sigmf(root, "multiple-tones", multiple_tones(duration=0.01), 100_000.0, "Tones")
            self.assertEqual(1_000, load_recording(root / "qpsk.sigmf-meta").sample_count)
            self.assertEqual(1_000, load_recording(root / "multiple-tones.sigmf-meta").sample_count)
            results = generate_segmented_results(root / "acoustic-events.json")
            self.assertTrue(results.is_file())


if __name__ == "__main__":
    unittest.main()
