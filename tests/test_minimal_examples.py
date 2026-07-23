import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

import plotly.graph_objects as go

from sigvue.profile import load_browser_profile
from sigvue.web.application import create_app
import sigvue_examples.style as style


ROOT = Path(__file__).resolve().parents[1]


class MinimalExampleTests(unittest.TestCase):
    def test_direct_runtime_dependencies_and_test_extra_are_declared(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        dependencies = tuple(project["dependencies"])
        for package in ("certifi", "numpy", "plotly", "scipy", "sigvue"):
            self.assertTrue(any(value.startswith(package) for value in dependencies))
        self.assertEqual(
            "sigvue>=2026.37",
            next(value for value in dependencies if value.startswith("sigvue")),
        )

    def test_profile_loads_copyable_comms_and_waterfall_workspaces(self):
        app = create_app(config_path=ROOT / "browser.toml")
        identifiers = [workspace["id"] for workspace in app.list_workspaces()]
        self.assertEqual(
            [
                "digital-comms",
                "downloaded-waterfall",
                "acoustic-events-segmented",
                "radio-astronomy-rfi",
                "lte-recordings",
                "lfm-sigmf",
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

    def test_every_workspace_declares_standard_discovery_columns(self):
        app = create_app(config_path=ROOT / "browser.toml")
        for workspace in app.list_workspaces():
            listing = app.browse_items(workspace["id"], {})
            self.assertEqual(
                ["date", "sample_rate", "rf_frequency"],
                [column["key"] for column in listing["columns"]],
                workspace["id"],
            )

    def test_shared_styles_keep_standard_grid_and_offer_quiet_heatmap_grid(self):
        figure = style.style_figure(go.Figure(), "light", "Example")
        self.assertEqual(style.GRID, figure.layout.xaxis.gridcolor)
        self.assertEqual(0.5, figure.layout.xaxis.gridwidth)
        self.assertEqual("rgba(96,113,125,0.12)", style.heatmap_grid_color("light"))
        self.assertEqual("rgba(169,189,194,0.13)", style.heatmap_grid_color("dark"))


if __name__ == "__main__":
    unittest.main()
