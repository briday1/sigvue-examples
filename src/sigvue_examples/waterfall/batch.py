"""Background report actions for the reusable waterfall workspace."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re
from typing import Callable
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sigvue.plugin import Batch, BatchDestination, BatchRequest, BatchResult, CapabilityChoice, DataResource

from ..io.sigmf.recording import SigMFRecording
from .domain import GroupedSigMFRecording, _waterfall_spectrogram


Recording = SigMFRecording | GroupedSigMFRecording


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "recording"


def _members(recording: Recording) -> tuple[tuple[str, SigMFRecording], ...]:
    if isinstance(recording, GroupedSigMFRecording):
        return tuple(zip(recording.labels, recording.recordings))
    return tuple(
        (f"Channel {index + 1}", recording)
        for index in range(recording.channel_count)
    )


def _report_figure(recording: Recording) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=(0.15, 0.85),
        vertical_spacing=0.06,
    )
    for index, (label, member) in enumerate(_members(recording)):
        count = min(member.sample_count, max(8, round(member.sample_rate * 0.002)))
        channel = 0 if isinstance(recording, GroupedSigMFRecording) else index
        samples = member.read(0, count)[channel]
        fft_size = min(4096, max(8, samples.size))
        waterfall, average, time_edges = _waterfall_spectrogram(
            samples,
            fft_size=fft_size,
            maximum_rows=1_000_000,
        )
        frequency = np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / member.sample_rate)) / 1e6
        captures = member.metadata.get("captures", [{}])
        center_mhz = float(captures[0].get("core:frequency", 0.0)) / 1e6 if captures else 0.0
        frequency += center_mhz
        visible = index == 0
        figure.add_trace(go.Scatter(
            x=frequency,
            y=average,
            name=f"{label} average PSD",
            visible=visible,
        ), row=1, col=1)
        figure.add_trace(go.Heatmap(
            x=frequency,
            y=time_edges / member.sample_rate * 1e3,
            z=waterfall,
            name=label,
            colorbar={"title": "dBFS"},
            colorscale="Plasma",
            visible=visible,
        ), row=2, col=1)
    buttons = []
    for member_index, (label, _) in enumerate(_members(recording)):
        visible = [False] * (2 * len(_members(recording)))
        visible[2 * member_index] = True
        visible[2 * member_index + 1] = True
        buttons.append({"label": label, "method": "update", "args": [{"visible": visible}]})
    figure.update_layout(
        title="Exact first 2 ms spectrum and waterfall",
        template="plotly_white",
        margin={"l": 70, "r": 35, "t": 70, "b": 55},
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1, "xanchor": "right", "y": 1.15}],
    )
    figure.update_yaxes(title_text="Power (dBFS)", row=1, col=1)
    figure.update_yaxes(title_text="Recording time (ms)", row=2, col=1)
    figure.update_xaxes(title_text="RF frequency (MHz)", row=2, col=1)
    return figure


def _write_report(resource: DataResource, recording: Recording, directory: Path) -> Path:
    target = directory / f"{_safe_name(resource.identifier)}-waterfall-report.html"
    plot = _report_figure(recording).to_html(full_html=False, include_plotlyjs="cdn")
    target.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>"
        f"{escape(resource.title)} report</title></head><body style='font-family:system-ui;margin:2rem'>"
        f"<h1>{escape(resource.title)}</h1><p>{escape(resource.subtitle or '')}</p>"
        f"<dl><dt>Sample rate</dt><dd>{recording.sample_rate:g} samples/s</dd>"
        f"<dt>Duration</dt><dd>{recording.duration_seconds:g} s</dd>"
        f"<dt>Members</dt><dd>{len(_members(recording))}</dd></dl>{plot}</body></html>",
        encoding="utf-8",
    )
    return target


class WaterfallBatch(Batch[Recording]):
    """Produce reports from catalog rows without opening interactive item pages."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    @property
    def item_actions(self) -> tuple[CapabilityChoice, ...]:
        return (
            CapabilityChoice("report", "Build waterfall report"),
            CapabilityChoice("metadata", "Export metadata JSON"),
        )

    @property
    def workspace_actions(self) -> tuple[CapabilityChoice, ...]:
        return (CapabilityChoice("report-all", "Build workspace report"),)

    def item_destination(self, resource: DataResource, request: BatchRequest) -> BatchDestination:
        directory = self.output_root / "items" / _safe_name(resource.identifier)
        filename = (
            f"{_safe_name(resource.identifier)}-waterfall-report.html"
            if request.action == "report"
            else f"{_safe_name(resource.identifier)}-metadata.json"
        )
        return BatchDestination(directory, (filename,), "Output already generated")

    def workspace_destination(
        self,
        resources: tuple[DataResource, ...],
        request: BatchRequest,
    ) -> BatchDestination:
        return BatchDestination(
            self.output_root / "workspace",
            ("waterfall-workspace-report.zip",),
            "Workspace report already generated",
        )

    def run_item(
        self,
        resource: DataResource,
        source_data: Recording,
        request: BatchRequest,
        directory: Path,
    ) -> BatchResult:
        if request.action == "report":
            report = _write_report(resource, source_data, directory)
            return BatchResult((report,), "Interactive waterfall report generated")
        target = directory / f"{_safe_name(resource.identifier)}-metadata.json"
        metadata = (
            {label: member.metadata for label, member in _members(source_data)}
            if isinstance(source_data, GroupedSigMFRecording)
            else source_data.metadata
        )
        target.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return BatchResult((target,), "SigMF metadata exported")

    def run_workspace(
        self,
        resources: tuple[DataResource, ...],
        open_resource: Callable[[DataResource], Recording],
        request: BatchRequest,
        directory: Path,
    ) -> BatchResult:
        reports = tuple(_write_report(resource, open_resource(resource), directory) for resource in resources)
        index = directory / "workspace-report.html"
        links = "".join(
            f"<li><a href='{escape(report.name)}'>{escape(resource.title)}</a></li>"
            for resource, report in zip(resources, reports)
        )
        index.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><title>Workspace report</title></head>"
            f"<body style='font-family:system-ui;margin:2rem'><h1>Waterfall workspace report</h1><ul>{links}</ul></body></html>",
            encoding="utf-8",
        )
        archive = directory / "waterfall-workspace-report.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
            bundle.write(index, index.name)
            for report in reports:
                bundle.write(report, report.name)
        return BatchResult((archive,), f"Generated {len(reports)} recording reports")


__all__ = ["WaterfallBatch"]
