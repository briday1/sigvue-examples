"""End-to-end coverage for the compact real-data workspace pipelines."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from sigvue_examples.ecg.workspace import create_workspace as create_ecg_workspace
from sigvue_examples.weather_radar.workspace import (
    create_workspace as create_weather_workspace,
)

from tests.test_weather_radar import synthetic_n0b
from tests.test_wfdb import (
    _annotation_word,
    _pack_format_212,
    _signed_checksum,
)


def _write_ecg_fixture(root: Path) -> Path:
    sample_count = 3_600
    samples = np.stack(
        (
            ((np.arange(sample_count) % 240) - 120) * 4,
            ((np.arange(sample_count) % 180) - 90) * 3,
        )
    ).astype(np.int16)
    checksums = tuple(_signed_checksum(channel) for channel in samples)
    header = root / "fixture.hea"
    header.write_text(
        "\n".join(
            (
                f"fixture 2 360 {sample_count}",
                (
                    "fixture.dat 212 200(0)/mV 12 0 "
                    f"{samples[0, 0]} {checksums[0]} 0 MLII"
                ),
                (f"fixture.dat 212 200(0)/mV 12 0 {samples[1, 0]} {checksums[1]} 0 V5"),
                "",
            )
        ),
        encoding="ascii",
    )
    (root / "fixture.dat").write_bytes(_pack_format_212(samples))
    annotations = b"".join(
        [_annotation_word(180, 1)]
        + [_annotation_word(360, 1) for _ in range(9)]
        + [_annotation_word(0, 0)]
    )
    (root / "fixture.atr").write_bytes(annotations)
    return header


class RealDataWorkspaceTests(unittest.TestCase):
    def test_ecg_workspace_renders_exact_samples_and_hoverable_annotations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_ecg_fixture(root)
            workspace = create_ecg_workspace({"data_root": root})
            opened = workspace.open_item(
                workspace.discover_items()[0].identifier,
            )
            waveform = opened.page.views[0].callback({})
            rr = opened.page.views[2].callback({})
            morphology = opened.page.views[3].callback({})

        self.assertTrue(workspace.lazy_views)
        self.assertEqual("windowed", opened.page.playback.mode)
        self.assertEqual("14.06 KiB", opened.page.statistics["Buffer memory"])
        self.assertEqual(3_600, len(waveform.data[0].x))
        self.assertEqual(
            "%{customdata}<extra></extra>",
            waveform.data[1].hovertemplate,
        )
        self.assertEqual(10, len(waveform.data[1].x))
        self.assertEqual(9, len(rr.data[0].x))
        self.assertEqual("Arithmetic mean", morphology.data[-1].name)
        self.assertEqual(10, morphology.data[-1].zorder)
        self.assertTrue(all(trace.zorder == 0 for trace in morphology.data[:-1]))

    def test_weather_workspace_uses_segmented_scans_without_radial_profile(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "TLX_N0B_2024_05_20_03_10_54"
            second = root / "TLX_N0B_2024_05_20_03_20_54"
            first.write_bytes(synthetic_n0b())
            second.write_bytes(synthetic_n0b(scan_seconds=12_054))
            workspace = create_weather_workspace({"data_root": root})
            resources = workspace.discover_items()
            opened = workspace.open_item(
                resources[0].identifier,
            )
            controls = {control.name: control for control in opened.page.controls}
            ppi = opened.page.views[0].callback(
                {
                    "weather_radar_ppi_range_km": "1.0",
                    "weather_radar_ppi_pixels": "256",
                    "weather_radar_colormap": "Viridis",
                }
            )
            advanced = workspace.open_item_with_values(
                resources[0].identifier,
                {
                    "__segment_id": second.name,
                    "weather_radar_colormap": "Turbo",
                },
            )

        self.assertTrue(workspace.lazy_views)
        self.assertEqual(1, len(resources))
        self.assertEqual("segmented", opened.page.playback.mode)
        self.assertEqual(2, len(opened.page.playback.segments))
        self.assertEqual(first.name, opened.page.playback.selected_segment_id)
        self.assertEqual(second.name, advanced.page.playback.selected_segment_id)
        self.assertEqual("2 of 2", advanced.page.statistics["Scan"])
        self.assertNotIn("weather_radar_radial_index", controls)
        colormap = controls["weather_radar_colormap"]
        self.assertEqual("colormap", colormap.control_type)
        self.assertEqual("NEXRAD", colormap.default)
        self.assertEqual("NEXRAD", colormap.options[0])
        self.assertEqual(11, len(colormap.options))
        self.assertEqual(11, len(colormap.option_previews))
        self.assertEqual(101, len(colormap.option_previews[0]))
        self.assertTrue(all(colormap.option_previews))
        self.assertEqual(
            [
                "weather-radar-ppi",
                "weather-radar-histogram",
                "weather-radar-statistics",
                "weather-radar-metadata",
            ],
            [view.name for view in opened.page.views],
        )
        self.assertEqual((256, 256), np.asarray(ppi.data[0].z).shape)
        self.assertEqual((-1.0, 1.0), tuple(ppi.layout.xaxis.range))
        self.assertIn("Buffer memory", opened.page.statistics)


if __name__ == "__main__":
    unittest.main()
