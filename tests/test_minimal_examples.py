import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

import plotly.graph_objects as go

from scripts.generate_comms import write_recording
from scripts.generate_segmented_results import generate as generate_events
from sigvue.profile import load_browser_profile
from sigvue.web.application import create_app
from sigvue_examples.events.workspace import (
    create_workspace as create_events_workspace,
)
import sigvue_examples.style as style


ROOT = Path(__file__).resolve().parents[1]


class MinimalExampleTests(unittest.TestCase):
    @staticmethod
    def dependency_names(requirements):
        return {
            re.split(r"[\s;<>=!~\[]", requirement, maxsplit=1)[0].lower()
            for requirement in requirements
        }

    def test_direct_runtime_dependencies_and_test_extra_are_declared(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        dependencies = tuple(project["dependencies"])
        self.assertEqual(
            {"certifi", "numpy", "plotly", "scipy", "sigvue"},
            self.dependency_names(dependencies),
        )
        self.assertEqual(
            "sigvue>=2026.38",
            next(value for value in dependencies if value.startswith("sigvue")),
        )
        extras = project["optional-dependencies"]
        self.assertEqual(
            {"pytest", "tomli"},
            self.dependency_names(extras["test"]),
        )
        self.assertEqual(
            {"build", "twine"},
            self.dependency_names(extras["release"]),
        )

    def test_profile_loads_copyable_comms_and_waterfall_workspaces(self):
        app = create_app(config_path=ROOT / "browser.toml")
        identifiers = [workspace["id"] for workspace in app.list_workspaces()]
        self.assertEqual(
            [
                "digital-comms",
                "mit-bih-ecg",
                "downloaded-waterfall",
                "acoustic-events-segmented",
                "radio-astronomy-rfi",
                "lte-recordings",
                "lfm-sigmf",
                "weather-radar",
            ],
            identifiers,
        )
        self.assertTrue(all(workspace.lazy_views for workspace in app.registry.list()))
        profile = load_browser_profile(ROOT / "browser.toml")
        waterfall_specs = [
            spec for spec in profile.workspaces
            if spec.module_name.endswith(".waterfall.workspace")
        ]
        self.assertEqual(3, len(waterfall_specs))
        self.assertEqual(1, len({(spec.module_name, spec.attribute) for spec in waterfall_specs}))

    def test_every_workspace_declares_relevant_discovery_columns(self):
        app = create_app(config_path=ROOT / "browser.toml")
        for workspace in app.list_workspaces():
            listing = app.browse_items(workspace["id"], {})
            keys = [column["key"] for column in listing["columns"]]
            self.assertTrue(keys, workspace["id"])
            self.assertEqual(len(keys), len(set(keys)), workspace["id"])
            if workspace["id"] not in {"mit-bih-ecg", "weather-radar"}:
                self.assertEqual(
                    ["date", "sample_rate", "rf_frequency"],
                    keys,
                    workspace["id"],
                )

    def test_event_workspace_default_matches_the_generated_data_location(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            generated = generate_events(
                root
                / "data"
                / "acoustic-events-segmented"
                / "acoustic-events.json"
            )
            workspace = create_events_workspace({"profile_dir": root})

            self.assertEqual(generated.parent.resolve(), workspace.source.directory)
            self.assertEqual(
                ["Synthetic acoustic event results"],
                [item.title for item in workspace.discover_items()],
            )

    def test_comms_generator_writes_sigmf_utc_datetime_with_z_suffix(self):
        with TemporaryDirectory() as directory:
            metadata_path, _ = write_recording(
                Path(directory),
                "QPSK",
                4,
                20260723,
            )
            timestamp = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )["captures"][0]["core:datetime"]

            self.assertTrue(timestamp.endswith("Z"))
            self.assertNotIn("+00:00", timestamp)

    def test_shared_styles_keep_standard_grid_and_offer_quiet_heatmap_grid(self):
        figure = style.style_figure(go.Figure(), "light", "Example")
        self.assertEqual(style.GRID, figure.layout.xaxis.gridcolor)
        self.assertEqual(0.5, figure.layout.xaxis.gridwidth)
        self.assertFalse(figure.layout.xaxis.automargin)
        self.assertFalse(figure.layout.yaxis.automargin)
        self.assertEqual("rgba(96,113,125,0.12)", style.heatmap_grid_color("light"))
        self.assertEqual("rgba(169,189,194,0.13)", style.heatmap_grid_color("dark"))


if __name__ == "__main__":
    unittest.main()
