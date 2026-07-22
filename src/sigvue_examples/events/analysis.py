"""Pass-through analysis contract for already processed acoustic products."""

from sigvue.plugin import Analysis

from .models import StoredEventResults


def process(event: StoredEventResults, settings: None) -> StoredEventResults:
    """The source already contains post-processed products, so preserve them."""
    return event


class EventAnalysis(Analysis[StoredEventResults, None, StoredEventResults]):
    def process(self, event: StoredEventResults, settings: None) -> StoredEventResults:
        return process(event, settings)


__all__ = ["EventAnalysis"]
