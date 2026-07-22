"""Annotation and export implementations for LFM collections."""

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.io import savemat

from sigvue.plugin import Annotation, AnnotationRequest, Annotator, Exporter, ExportRequest

from ..io.sigmf.capabilities import (
    FORMATS, SCOPES, add_sigmf_annotation, read_sigmf_annotations,
    waterfall_annotation_fields,
)
from ..io.sigmf.recording import load_recording
from .domain import LfmCollection, LfmInput


class LfmAnnotator(Annotator[LfmCollection, LfmInput]):
    """Store matching standard SigMF annotations on all four OTA members."""

    timeline_color_control = "lfm_annotation_region_color"

    @property
    def fields(self):
        return waterfall_annotation_fields(
            "waterfall-domain-1",
            time_scale=1.0,
            frequency_scale=1.0,
            time_offset_source="playback",
        )

    def discover(self, collection: LfmCollection):
        return read_sigmf_annotations(load_recording(collection.members["ota"][0].metadata_path))

    def annotate(self, collection: LfmCollection, delivered: LfmInput, request: AnnotationRequest) -> Annotation:
        try:
            start_seconds = float(request.values["start_seconds"])
            stop_seconds = float(request.values["stop_seconds"])
            frequency_lower_hz = float(request.values["frequency_lower_hz"])
            frequency_upper_hz = float(request.values["frequency_upper_hz"])
        except (KeyError, ValueError) as error:
            raise ValueError("Waterfall annotation bounds must be numeric") from error
        if start_seconds < 0 or stop_seconds <= start_seconds:
            raise ValueError("Annotation stop time must be after its non-negative start time")
        available = collection.sample_count("ota")
        start_sample = min(available, round(start_seconds * collection.sample_rate))
        stop_sample = min(available, round(stop_seconds * collection.sample_rate))
        if stop_sample <= start_sample:
            raise ValueError("Annotation time bounds do not contain any recording samples")
        identifier = str(uuid4())
        result = None
        for member in collection.members["ota"]:
            result = add_sigmf_annotation(
                load_recording(member.metadata_path),
                start_sample,
                stop_sample - start_sample,
                request,
                identifier=identifier,
                frequency_lower_hz=frequency_lower_hz,
                frequency_upper_hz=frequency_upper_hz,
            )
        assert result is not None
        return result

class LfmExporter(Exporter[LfmCollection, LfmInput]):
    """Serialize either the delivered OTA window or every collection member."""

    @property
    def scopes(self):
        return SCOPES

    @property
    def formats(self):
        return FORMATS

    def export(self, collection: LfmCollection, delivered: LfmInput, request: ExportRequest, directory: Path) -> Path:
        stem = collection.collection_path.stem if collection.collection_path else "lfm-collection"
        if request.scope == "buffer":
            start = delivered.start_sample
            arrays = {
                "calibration": delivered.calibration_counts,
                "terminated_noise": delivered.noise_counts,
                "ota": delivered.ota_counts,
            }
        else:
            start = 0
            arrays = {role.replace("-", "_"): collection.read(role) for role in collection.members}
        ota_count = arrays["ota"].shape[-1]
        target = directory / (
            f"{stem}-t{start / collection.sample_rate:.9f}s-"
            f"{ota_count / collection.sample_rate:.9f}s-{request.scope}.{request.format}"
        )
        metadata = {
            "sample_rate": collection.sample_rate,
            "start_sample": start,
            "scope": request.scope,
            "calibration_dbm": collection.calibration_dbm,
            "adc_bits": collection.adc_bits,
            "ota_prf_hz": collection.ota_prf_hz,
            "control_values": dict(request.control_values),
        }
        if request.format == "mat":
            mat_metadata = {**metadata, "control_values": json.dumps(metadata["control_values"], default=str)}
            savemat(target, {**mat_metadata, **arrays})
            return target
        if request.format != "json":
            raise ValueError(f"Unsupported LFM export format: {request.format}")
        with target.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata)[:-1])
            stream.write(', "samples": {')
            for index, (role, samples) in enumerate(arrays.items()):
                if index:
                    stream.write(",")
                stream.write(f'{json.dumps(role)}: {{"real": ')
                _write_numeric_matrix(stream, samples.real)
                stream.write(', "imag": ')
                _write_numeric_matrix(stream, samples.imag)
                stream.write("}")
            stream.write("}}")
        return target

def _write_numeric_matrix(stream: Any, values: np.ndarray, chunk_size: int = 16_384) -> None:
    stream.write("[")
    for channel_index, channel in enumerate(values):
        if channel_index:
            stream.write(",")
        stream.write("[")
        for start in range(0, channel.size, chunk_size):
            if start:
                stream.write(",")
            stream.write(",".join(format(float(value), ".9g") for value in channel[start : start + chunk_size]))
        stream.write("]")
    stream.write("]")


__all__ = ["LfmAnnotator", "LfmExporter"]
