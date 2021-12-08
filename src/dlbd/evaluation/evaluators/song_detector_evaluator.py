from abc import abstractmethod

import pandas as pd
from plotnine import aes, element_text, geom_line, ggplot, ggtitle, theme, theme_classic

from mouffet.utils.common import (
    deep_dict_update,
    expand_options_dict,
    listdict2dictlist,
)

from mouffet.evaluation.evaluator import Evaluator


class SongDetectorEvaluator(Evaluator):

    EVENTS_COLUMNS = {
        "index": "event_id",
        "event_index": "event_index",
        "recording_id": "recording_id",
        "start": "event_start",
        "end": "event_end",
        "event_duration": "event_duration",
    }
    TAGS_COLUMNS_RENAME = {"id": "tag_id"}

    DEFAULT_ACTIVITY_THRESHOLD = 0.85

    DEFAULT_PR_CURVE_OPTIONS = {
        "variable": "activity_threshold",
        "values": {"end": 1, "start": 0, "step": 0.05},
    }
