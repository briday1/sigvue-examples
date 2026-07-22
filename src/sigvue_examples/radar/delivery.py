"""Live/seek and whole-file UI delivery policies for LFM collections."""

from sigvue.plugin import Delivery, DeliveryContext, PlaybackMode

from ..io.sigmf.capabilities import read_sigmf_annotations
from ..io.sigmf.recording import load_recording
from .domain import LfmCollection, LfmInput


class BufferedDelivery(Delivery[LfmCollection, LfmInput]):
    """Framework policy for playback: deliver one requested OTA window."""

    def __init__(self, *, playback_mode: PlaybackMode = "live") -> None:
        if playback_mode not in {"seek", "live"}:
            raise ValueError("Buffered playback mode must be 'seek' or 'live'")
        self.playback_mode = playback_mode

    def prepare(self, collection: LfmCollection, ui: DeliveryContext) -> LfmInput:
        default_pri = 1 / collection.ota_prf_hz
        buffer_seconds = ui.number("buffer_seconds", default=0.02, minimum=default_pri, maximum=0.1, step=default_pri)
        processing_pri_seconds = ui.number(
            "processing_pri_seconds",
            label="Processing PRI (s)",
            default=default_pri,
            minimum=8 / collection.sample_rate,
            maximum=1.0,
            step=default_pri / 10,
        )
        seek_seconds = ui.number("seek_seconds", default=0.01, minimum=0.001, step=0.001)
        refresh_seconds = ui.number("refresh_seconds", default=0.15, minimum=0.05, step=0.05)
        available = collection.sample_count("ota")
        size = min(available, max(1, round(buffer_seconds * collection.sample_rate)))
        pri = min(size, max(8, round(collection.sample_rate * processing_pri_seconds)))
        duration = max(0.0, (available - size) / collection.sample_rate)
        time = ui.playback(
            mode=self.playback_mode,
            duration=duration,
            step=seek_seconds,
            refresh_interval=refresh_seconds,
            loop=False,
        )
        start = min(round(time * collection.sample_rate), available - size)
        return _input(collection, start=start, count=size, pri=pri, ui=ui)

class WholeFileDelivery(Delivery[LfmCollection, LfmInput]):
    """Framework policy for batch mode: deliver the complete OTA member files."""

    def __init__(self, *, default_processing_pri_seconds: float | None = None) -> None:
        self.default_processing_pri_seconds = default_processing_pri_seconds

    def prepare(self, collection: LfmCollection, ui: DeliveryContext) -> LfmInput:
        ui.playback(mode="static")
        default_pri_seconds = self.default_processing_pri_seconds or 1 / collection.ota_prf_hz
        processing_pri_seconds = ui.number(
            "processing_pri_seconds",
            label="Processing PRI (s)",
            default=default_pri_seconds,
            minimum=8 / collection.sample_rate,
            maximum=1.0,
            step=default_pri_seconds / 10,
        )
        pri = max(8, round(collection.sample_rate * processing_pri_seconds))
        return _input(collection, start=0, count=collection.sample_count("ota"), pri=pri, ui=ui)

def _input(collection: LfmCollection, *, start: int, count: int, pri: int, ui: DeliveryContext) -> LfmInput:
    calibration = ui.once("lfm-calibration-counts", lambda: collection.read("calibration"))
    noise = ui.once("lfm-noise-counts", lambda: collection.read("terminated-noise"))
    annotation_path = collection.members["ota"][0].metadata_path
    current_annotations = read_sigmf_annotations(load_recording(annotation_path)) if annotation_path.is_file() else ()
    return LfmInput(
        sample_rate=collection.sample_rate,
        calibration_dbm=collection.calibration_dbm,
        adc_bits=collection.adc_bits,
        pri_samples=pri,
        start_sample=start,
        calibration_counts=calibration,
        noise_counts=noise,
        ota_counts=collection.read("ota", start, count),
        annotations=current_annotations,
    )


__all__ = ["BufferedDelivery", "LfmInput", "WholeFileDelivery"]
