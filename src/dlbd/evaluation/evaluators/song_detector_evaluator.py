from mouffet.evaluation import Evaluator


class SongDetectorEvaluator(Evaluator):

    EVENTS_COLUMNS = {
        "index": "event_id",
        # "event_index": "event_index",
        "recording_id": "recording_id",
        "start": "event_start",
        "end": "event_end",
        "event_duration": "event_duration",
    }
    TAGS_COLUMNS_RENAME = {"id": "tag_id"}

    DEFAULT_ACTIVITY_THRESHOLD = 0.9

    DEFAULT_PR_CURVE_OPTIONS = {
        "variable": "activity_threshold",
        "values": {"end": 1, "start": 0, "step": 0.05},
    }
