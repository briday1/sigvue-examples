import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.io import loadmat

from sigvue.plugin import AnnotationRequest, ExportRequest
from sigvue_examples.capabilities import SigMFAnnotator, SigMFExporter, WaterfallSigMFAnnotator
from sigvue_examples.sigmf import load_recording


class SigMFCapabilityTests(unittest.TestCase):
    def make_recording(self, root: Path):
        metadata_path = root / "capture.sigmf-meta"
        metadata_path.write_text(json.dumps({
            "global": {"core:datatype": "ci16_le", "core:sample_rate": 1_000.0, "core:version": "1.2.6"},
            "captures": [{"core:sample_start": 0}],
            "annotations": [],
        }), encoding="utf-8")
        iq = np.column_stack((np.arange(8, dtype=np.int16), -np.arange(8, dtype=np.int16)))
        iq.astype("<i2").tofile(root / "capture.sigmf-data")
        return load_recording(metadata_path)

    def test_annotation_uses_standard_sigmf_fields_and_is_rediscovered(self):
        with TemporaryDirectory() as directory:
            recording = self.make_recording(Path(directory))
            delivered = type("Window", (), {"start_sample": 2, "samples": recording.read(2, 3)})()
            annotator = SigMFAnnotator()
            created = annotator.annotate(
                recording,
                delivered,
                AnnotationRequest(0.0, values={"comment": "Interesting burst"}),
            )
            entry = json.loads(recording.metadata_path.read_text())["annotations"][0]
            self.assertEqual(2, entry["core:sample_start"])
            self.assertEqual(3, entry["core:sample_count"])
            self.assertNotIn("core:label", entry)
            self.assertEqual("Interesting burst", entry["core:comment"])
            self.assertEqual("Sigvue Examples", entry["core:generator"])
            self.assertEqual(created.identifier, entry["core:uuid"])
            self.assertIsNone(created.label)
            self.assertEqual((created,), annotator.discover(recording))

    def test_exporter_owns_buffer_and_full_json_and_mat_serialization(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            recording = self.make_recording(root)
            delivered = type("Window", (), {"start_sample": 2, "samples": recording.read(2, 3)})()
            exporter = SigMFExporter()
            json_path = exporter.export(recording, delivered, ExportRequest("buffer", "json"), root)
            payload = json.loads(json_path.read_text())
            self.assertEqual(2, payload["start_sample"])
            self.assertEqual(3, payload["sample_count"])
            self.assertEqual(3, len(payload["samples"]["real"][0]))
            mat_path = exporter.export(recording, delivered, ExportRequest("full", "mat"), root)
            payload = loadmat(mat_path)
            self.assertEqual((1, 8), payload["samples"].shape)
            self.assertEqual(0, int(payload["start_sample"][0, 0]))

    def test_waterfall_annotation_writes_time_frequency_bounds(self):
        with TemporaryDirectory() as directory:
            recording = self.make_recording(Path(directory))
            delivered = type("Window", (), {"start_sample": 0, "samples": recording.read(0, 8)})()
            annotator = WaterfallSigMFAnnotator("spectrum", "annotation_region_color")
            self.assertEqual("annotation_region_color", annotator.timeline_color_control)
            self.assertTrue(
                all(
                    field.plot_binding.selection_policy == "box_preferred"
                    for field in annotator.fields
                    if field.plot_binding is not None
                )
            )
            created = annotator.annotate(
                recording,
                delivered,
                AnnotationRequest(0.0, values={
                    "start_seconds": "0.002",
                    "stop_seconds": "0.006",
                    "frequency_lower_hz": "100",
                    "frequency_upper_hz": "200",
                    "comment": "Bounded signal",
                }),
            )
            entry = json.loads(recording.metadata_path.read_text())["annotations"][0]
            self.assertEqual((2, 4), (entry["core:sample_start"], entry["core:sample_count"]))
            self.assertEqual((100.0, 200.0), (entry["core:freq_lower_edge"], entry["core:freq_upper_edge"]))
            self.assertEqual((100.0, 200.0), (created.frequency_lower_hz, created.frequency_upper_hz))
            fields = {field.name: field for field in annotator.fields}
            self.assertEqual("xaxis2", fields["frequency_lower_hz"].plot_binding.axis)
            self.assertEqual("yaxis2", fields["start_seconds"].plot_binding.axis)


if __name__ == "__main__":
    unittest.main()
