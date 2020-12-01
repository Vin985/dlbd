import datetime
import functools
import math
from functools import partial

import numpy as np
import pandas as pd

from ..lib.detector import Detector


class SubsamplingDetector(Detector):

    DEFAULT_ISOLATE_EVENTS = True
    DEFAULT_EVENT_THRESHOLD = 0.5

    def resample_max(self, x, threshold=0.98):
        if max(x) >= threshold:
            return 2
        return 0

    def has_tag(self, x):
        if max(x) > 0:
            return 1
        return 0

    def get_tag_index(self, x, step, tags):
        start = x.index.values[0].item() / 10 ** 9
        end = start + step / 1000
        tmp = np.unique(tags["id"][(tags["start"] < end) & (tags["end"] >= start)])
        return tmp.tolist()

    def isolate_events(self, predictions, step):
        tmp = predictions.loc[predictions.event > 0]
        tmp.reset_index(inplace=True)

        step = datetime.timedelta(milliseconds=step)
        start = None
        prev_time = 0
        events = []
        event_id = 1
        recording_id = None
        if not tmp.empty:
            for _, x in tmp.iterrows():
                if not recording_id:
                    recording_id = x.recording_id
                if not start:
                    prev_time = x.datetime
                    start = prev_time
                    continue
                diff = x.datetime - prev_time
                if diff > step:
                    end = prev_time + step
                    events.append(
                        {
                            "event_id": event_id,
                            "recording_id": x.recording_id,
                            "start": start.timestamp(),
                            "end": end.timestamp(),
                        }
                    )
                    event_id += 1
                    start = x.datetime
                prev_time = x.datetime

            end = prev_time + step
            events.append(
                {
                    "event_id": event_id,
                    "recording_id": recording_id,
                    "start": start.timestamp(),
                    "end": end.timestamp(),
                }
            )

        events = pd.DataFrame(events)
        return events

    def get_events(self, predictions, tags, options):
        events = predictions.groupby(
            "recording_id", as_index=True, observed=True
        ).apply(self.get_recording_events, tags, options)
        return events

    @staticmethod
    def list_append(x, y):
        x.append(y)

    def get_recording_events(self, predictions, tags, options=None):
        options = options or {}
        tags = tags[tags.recording_id == predictions.name]
        threshold = options.get("event_threshold", self.DEFAULT_EVENT_THRESHOLD)
        predictions.loc[predictions.activity > threshold, "event"] = 2
        if not predictions.empty:
            step = predictions.time.values[1] - predictions.time.values[0]
            for tag in tags.itertuples():
                start = math.ceil(tag.tag_start / step)
                end = math.ceil(tag.tag_end / step)
                f = partial(self.list_append, y=tag.id)
                predictions.tag_index[start:end].map(f)
        return predictions

    def match_events_apply(self, predictions, tags, options):
        recording_id = predictions.name
        current_tags = tags.loc[tags.recording_id == recording_id]
        min_activity = options.get("min_activity", self.DEFAULT_MIN_ACTIVITY)
        step = options.get("sample_step", self.DEFAULT_MIN_DURATION) * 1000
        resampler = predictions.resample(str(step) + "ms") if step else predictions
        resample_func = functools.partial(self.resample_max, threshold=min_activity)
        tmp_tags = {
            "id": current_tags.id.to_numpy().astype(str),
            "start": current_tags.tag_start.to_numpy(),
            "end": current_tags.tag_end.to_numpy(),
        }
        tag_func = functools.partial(self.get_tag_index, step=step, tags=tmp_tags)
        res = resampler.agg({"activity": resample_func, "tag_index": tag_func}).rename(
            columns={"activity": "event"}
        )
        # res["recording_id"] = recording_id
        return res

    def get_stats(self, df, tags):
        # TODO: make stats coherent between no resampling and resampling: use tag index in the same way
        res = df.tag + df.event

        n_true_positives = len(res[res == 3])
        n_true_negatives = len(res[res == 0])
        n_false_positives = len(res[res == 2])
        n_false_negatives = len(res[res == 1])

        matched_tags = set(
            df.tag_index.loc[(df.tag_index.str.len() > 0) & (df.event > 0)].sum()
        )

        n_tags = tags.shape[0]
        n_unmatched_tags = n_tags - len(matched_tags)

        precision = round(n_true_positives / (n_true_positives + n_false_positives), 3)
        recall = round(n_true_positives / (n_true_positives + n_false_negatives), 3)
        recall2 = round(n_true_positives / (n_true_positives + n_unmatched_tags), 3)
        f1_score = round(2 * precision * recall / (precision + recall), 3)
        return {
            "n_events": int(df.event.sum() / 2),
            "n_tags": n_tags,
            "n_true_positives": n_true_positives,
            "n_false_positives": n_false_positives,
            "n_true_negatives": n_true_negatives,
            "n_false_negatives": n_false_negatives,
            "n_tags_matched": len(matched_tags),
            "n_tags_unmatched": n_unmatched_tags,
            "precision": precision,
            "recall": recall,
            "recall2": recall2,
            "f1_score": f1_score,
        }

    def evaluate(self, predictions, tags, options):
        preds = predictions.copy()
        # preds.loc[:, "tag_index"] = ""
        tags = tags["tags_df"]
        preds.loc[:, "tag_index"] = pd.Series([[] for i in range(preds.shape[0])])
        preds.loc[:, "event"] = 0
        preds.recording_id = preds.recording_id.astype("category")
        step = options.get("sample_step", self.DEFAULT_MIN_DURATION) * 1000

        if step:
            preds.loc[:, "datetime"] = pd.to_datetime(preds.time * 10 ** 9)
            preds.set_index("datetime", inplace=True)
            apply_func = self.match_events_apply
        else:
            apply_func = self.get_recording_events

        events = (
            preds.groupby("recording_id", as_index=False, observed=True)
            .apply(apply_func, tags=tags, options=options)
            .reset_index()
        )
        events.loc[:, "tag"] = 0
        events.loc[events.tag_index.str.len() > 0, "tag"] = 1

        stats = self.get_stats(events, tags)
        print("Stats for options {0}: {1}".format(options, stats))
        return {"options": options, "stats": stats, "matches": events}
