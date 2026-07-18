"""SigMF adapters for Sigvue's optional annotation and export capabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.io import savemat

from sigvue.plugin import (
    Annotation,
    AnnotationField,
    AnnotationPlotBinding,
    AnnotationRequest,
    CapabilityChoice,
    DataAnnotator,
    DataExporter,
    ExportRequest,
)

from .sigmf import SigMFRecording, annotations, append_annotation


SCOPES = (CapabilityChoice("buffer", "Current buffer"), CapabilityChoice("full", "Full file"))
FORMATS = (CapabilityChoice("json", "JSON"), CapabilityChoice("mat", "MAT"))


def annotation_fields() -> tuple[AnnotationField, ...]:
    return (AnnotationField("comment", "Description / comment", "textarea", required=True),)


def read_sigmf_annotations(recording: SigMFRecording) -> tuple[Annotation, ...]:
    result = []
    for index, entry in enumerate(annotations(recording.metadata_path)):
        start = int(entry["core:sample_start"])
        count = entry.get("core:sample_count")
        result.append(Annotation(
            identifier=str(entry.get("core:uuid") or f"{start}:{index}"),
            start_seconds=start / recording.sample_rate,
            duration_seconds=None if count is None else int(count) / recording.sample_rate,
            label=str(entry["core:label"]) if entry.get("core:label") else None,
            comment=str(entry.get("core:comment") or "") or None,
            frequency_lower_hz=(
                float(entry["core:freq_lower_edge"])
                if entry.get("core:freq_lower_edge") is not None
                else None
            ),
            frequency_upper_hz=(
                float(entry["core:freq_upper_edge"])
                if entry.get("core:freq_upper_edge") is not None
                else None
            ),
        ))
    return tuple(result)


def add_sigmf_annotation(
    recording: SigMFRecording,
    start_sample: int,
    sample_count: int,
    request: AnnotationRequest,
    *,
    identifier: str | None = None,
    frequency_lower_hz: float | None = None,
    frequency_upper_hz: float | None = None,
) -> Annotation:
    annotation_id = identifier or str(uuid4())
    comment = request.values.get("comment", "").strip()
    if not comment:
        raise ValueError("An annotation description/comment is required")
    entry: dict[str, object] = {
        "core:sample_start": max(0, int(start_sample)),
        "core:sample_count": max(0, int(sample_count)),
        "core:comment": comment,
        "core:generator": "Sigvue Examples",
        "core:uuid": annotation_id,
    }
    if (frequency_lower_hz is None) != (frequency_upper_hz is None):
        raise ValueError("Both lower and upper annotation frequencies are required")
    if frequency_lower_hz is not None and frequency_upper_hz is not None:
        if frequency_lower_hz >= frequency_upper_hz:
            raise ValueError("Lower annotation frequency must be below upper annotation frequency")
        entry["core:freq_lower_edge"] = float(frequency_lower_hz)
        entry["core:freq_upper_edge"] = float(frequency_upper_hz)
    append_annotation(recording.metadata_path, entry)
    return Annotation(
        annotation_id,
        entry["core:sample_start"] / recording.sample_rate,
        entry["core:sample_count"] / recording.sample_rate,
        None,
        comment,
        frequency_lower_hz,
        frequency_upper_hz,
    )


def waterfall_annotation_fields(
    view: str,
    *,
    time_scale: float = 1e-3,
    frequency_scale: float = 1e6,
    time_offset_source: str = "none",
) -> tuple[AnnotationField, ...]:
    """Inputs populated from the currently visible waterfall axes."""
    return (
        AnnotationField(
            "start_seconds",
            "Recording start (s)",
            "number",
            required=True,
            plot_binding=AnnotationPlotBinding(
                view, "yaxis2", "lower", scale=time_scale, offset_source=time_offset_source
            ),
        ),
        AnnotationField(
            "stop_seconds",
            "Recording stop (s)",
            "number",
            required=True,
            plot_binding=AnnotationPlotBinding(
                view, "yaxis2", "upper", scale=time_scale, offset_source=time_offset_source
            ),
        ),
        AnnotationField(
            "frequency_lower_hz",
            "Lower RF frequency (Hz)",
            "number",
            required=True,
            plot_binding=AnnotationPlotBinding(view, "xaxis2", "lower", scale=frequency_scale),
        ),
        AnnotationField(
            "frequency_upper_hz",
            "Upper RF frequency (Hz)",
            "number",
            required=True,
            plot_binding=AnnotationPlotBinding(view, "xaxis2", "upper", scale=frequency_scale),
        ),
        AnnotationField("comment", "Description / comment", "textarea", required=True),
    )


class WaterfallSigMFAnnotator(DataAnnotator[SigMFRecording, Any]):
    """SigMF annotator whose bounds come from one waterfall Plotly view."""

    def __init__(self, view: str, timeline_color_control: str):
        self.view = view
        self.timeline_color_control = timeline_color_control

    @property
    def fields(self) -> tuple[AnnotationField, ...]:
        return waterfall_annotation_fields(self.view)

    def discover(self, recording: SigMFRecording) -> tuple[Annotation, ...]:
        return read_sigmf_annotations(recording)

    def annotate(self, recording: SigMFRecording, delivered: Any, request: AnnotationRequest) -> Annotation:
        try:
            start_seconds = float(request.values["start_seconds"])
            stop_seconds = float(request.values["stop_seconds"])
            frequency_lower_hz = float(request.values["frequency_lower_hz"])
            frequency_upper_hz = float(request.values["frequency_upper_hz"])
        except (KeyError, ValueError) as error:
            raise ValueError("Waterfall annotation bounds must be numeric") from error
        if start_seconds < 0 or stop_seconds <= start_seconds:
            raise ValueError("Annotation stop time must be after its non-negative start time")
        start_sample = min(recording.sample_count, round(start_seconds * recording.sample_rate))
        stop_sample = min(recording.sample_count, round(stop_seconds * recording.sample_rate))
        if stop_sample <= start_sample:
            raise ValueError("Annotation time bounds do not contain any recording samples")
        return add_sigmf_annotation(
            recording,
            start_sample,
            stop_sample - start_sample,
            request,
            frequency_lower_hz=frequency_lower_hz,
            frequency_upper_hz=frequency_upper_hz,
        )


class SigMFAnnotator(DataAnnotator[SigMFRecording, Any]):
    @property
    def fields(self) -> tuple[AnnotationField, ...]:
        return annotation_fields()

    def discover(self, recording: SigMFRecording) -> tuple[Annotation, ...]:
        return read_sigmf_annotations(recording)

    def annotate(self, recording: SigMFRecording, delivered: Any, request: AnnotationRequest) -> Annotation:
        start = int(getattr(delivered, "start_sample", round(request.position_seconds * recording.sample_rate)))
        samples = np.asarray(getattr(delivered, "samples", np.empty((recording.channel_count, 0))))
        count = int(samples.shape[-1]) if samples.ndim else 0
        return add_sigmf_annotation(recording, start, count, request)


def _filename(stem: str, start: int, count: int, rate: float, scope: str, extension: str) -> str:
    start_seconds = start / rate
    duration_seconds = count / rate
    return f"{stem}-t{start_seconds:.9f}s-{duration_seconds:.9f}s-{scope}.{extension}"


def write_sample_export(
    directory: Path,
    *,
    stem: str,
    samples: np.ndarray,
    sample_rate: float,
    start_sample: int,
    scope: str,
    format: str,
    metadata: dict[str, object],
    control_values: dict[str, object] | None = None,
) -> Path:
    samples = np.asarray(samples)
    target = directory / _filename(stem, start_sample, samples.shape[-1], sample_rate, scope, format)
    common = {
        "sample_rate": sample_rate,
        "start_sample": start_sample,
        "sample_count": samples.shape[-1],
        "channel_count": samples.shape[0],
        "scope": scope,
        "metadata": metadata,
        "control_values": control_values or {},
    }
    if format == "mat":
        mat_common = {
            **{key: value for key, value in common.items() if key not in {"metadata", "control_values"}},
            "metadata_json": json.dumps(metadata, default=str),
            "control_values_json": json.dumps(control_values or {}, default=str),
        }
        savemat(target, {**mat_common, "samples": samples})
        return target
    if format != "json":
        raise ValueError(f"Unsupported sample export format: {format}")
    with target.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(common, default=str)[:-1])
        stream.write(', "samples": {"real": ')
        _write_numeric_matrix(stream, samples.real)
        stream.write(', "imag": ')
        _write_numeric_matrix(stream, samples.imag)
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


class SigMFExporter(DataExporter[SigMFRecording, Any]):
    @property
    def scopes(self) -> tuple[CapabilityChoice, ...]:
        return SCOPES

    @property
    def formats(self) -> tuple[CapabilityChoice, ...]:
        return FORMATS

    def export(self, recording: SigMFRecording, delivered: Any, request: ExportRequest, directory: Path) -> Path:
        if request.scope == "full":
            start, samples = 0, recording.read(0, recording.sample_count)
        else:
            start = int(getattr(delivered, "start_sample"))
            samples = np.asarray(getattr(delivered, "samples"))
        return write_sample_export(
            directory,
            stem=recording.metadata_path.name.removesuffix(".sigmf-meta"),
            samples=samples,
            sample_rate=recording.sample_rate,
            start_sample=start,
            scope=request.scope,
            format=request.format,
            metadata=recording.metadata,
            control_values=dict(request.control_values),
        )
