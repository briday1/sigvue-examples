"""Workspace UI for annotated ECG products."""

from sigvue.helpers import format_bytes
from sigvue.plugin import ViewContext

from ..style import ORANGE, TEAL, style_figure
from .models import ECGProducts
from .plots import morphology_figure, rr_figure, waveform_figure


def present(products: ECGProducts, ui: ViewContext) -> None:
    """Declare lazy ECG views and their appearance controls."""
    waveform_style = ui.trace_style(
        "ecg_waveform",
        label="ECG waveform",
        color=TEAL,
        width=1.2,
    )
    annotation_style = ui.trace_style(
        "reference_annotations",
        label="Reference annotations",
        color=ORANGE,
        width=2.0,
        marker="diamond",
        opacity=0.9,
        group="Annotations",
    )
    ui.stat("Record", products.recording.record_name)
    ui.stat("Window start", f"{products.time_seconds[0]:.3f} s")
    ui.stat("Window width", f"{products.duration_seconds:.3f} s")
    ui.stat("Native samples", f"{products.physical_samples.size:,}")
    ui.stat("Annotations in window", f"{len(products.annotations):,}")
    ui.stat("Beats in window", f"{products.beat_count:,}")
    ui.stat("Buffer memory", format_bytes(products.buffer_nbytes))

    lead_views = {
        name: (
            lambda index=index, name=name: style_figure(
                waveform_figure(
                    products,
                    index,
                    waveform_style,
                    annotation_style,
                ),
                ui.theme,
                f"MIT-BIH {products.recording.record_name} · {name}",
            )
        )
        for index, name in enumerate(products.recording.channel_names)
    }
    morphology_views = {
        name: (
            lambda index=index, name=name: style_figure(
                morphology_figure(products, index),
                ui.theme,
                f"Annotated beat morphology · {name}",
            )
        )
        for index, name in enumerate(products.recording.channel_names)
    }
    with ui.tab("ECG"):
        ui.view_switcher(
            "Lead",
            lead_views,
            key="ecg-lead",
            selector="dropdown",
            axis_navigation="bounded",
        )
    with ui.tab("RR intervals"):
        ui.plot(
            lambda: style_figure(
                rr_figure(products),
                ui.theme,
                "Reference-annotation RR intervals",
            ),
            key="ecg-rr",
            axis_navigation="bounded",
        )
    with ui.tab("Beat morphology"):
        ui.view_switcher(
            "Lead",
            morphology_views,
            key="ecg-morphology-lead",
            selector="dropdown",
            axis_navigation="bounded",
        )
    with ui.tab("Annotations"):
        ui.table(products.annotation_rows, key="ecg-annotations")
    with ui.tab("Metadata", update="static"):
        ui.table(products.metadata_rows, key="ecg-metadata")


__all__ = ["present"]
