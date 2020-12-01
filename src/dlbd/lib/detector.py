# from pandarallel import pandarallel

from abc import ABC, abstractmethod


class Detector(ABC):

    EVENTS_COLUMNS = {
        "index": "event_id",
        "event_index": "event_index",
        "recording_id": "recording_id",
        "start": "event_start",
        "end": "event_end",
        "event_duration": "event_duration",
    }
    TAGS_COLUMNS_RENAME = {"id": "tag_id"}

    DEFAULT_MIN_ACTIVITY = 0.85
    DEFAULT_MIN_DURATION = 0.1
    DEFAULT_END_THRESHOLD = 0.6

    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, predictions, tags, options):
        pass

