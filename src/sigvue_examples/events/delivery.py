"""UI delivery policy for selecting one precomputed acoustic event."""

from sigvue.plugin import Delivery, DeliveryContext, Segment

from .models import AcousticEventCollection, StoredEventResults


class StoredEventDelivery(Delivery[AcousticEventCollection, StoredEventResults]):
    def prepare(self, collection: AcousticEventCollection, ui: DeliveryContext) -> StoredEventResults:
        selected = ui.segmented(
            duration=collection.duration_seconds,
            segments=tuple(
                Segment(event.identifier, event.start_seconds, event.duration_seconds, event.label)
                for event in collection.events
            ),
        )
        return next(event for event in collection.events if event.identifier == selected.identifier)

__all__ = ["StoredEventDelivery", "StoredEventResults"]
