"""UI presentation for stored acoustic-event products."""

from sigvue.plugin import Presentation, ViewContext

from ..style import style_figure
from ..memory import format_bytes
from .models import StoredEventResults
from .plots import spectrum_figure, waveform_figure


def present(event: StoredEventResults, ui: ViewContext) -> None:
    ui.stat("Stored event", event.label)
    ui.stat("Confidence", f"{event.confidence:.1%}")
    ui.stat("Start", f"{event.start_seconds:.3f} s")
    ui.stat("Duration", f"{event.duration_seconds:.3f} s")
    ui.stat("Buffer memory", format_bytes(event.buffer_nbytes))
    with ui.tab("Stored waveform"):
        ui.plot(
            lambda: style_figure(waveform_figure(event), ui.theme, event.label),
            key="waveform",
        )
    with ui.tab("Stored spectrum"):
        ui.plot(
            lambda: style_figure(
                spectrum_figure(event), ui.theme, f"{event.label} spectrum",
            ),
            key="spectrum",
        )


class EventPresentation(Presentation[StoredEventResults]):
    def present(self, event: StoredEventResults, ui: ViewContext) -> None:
        present(event, ui)
