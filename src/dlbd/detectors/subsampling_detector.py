import datetime
import functools

import numpy as np
import pandas as pd

from .detector import Detector
import math


class SubsamplingDetector(Detector):

    DEFAULT_ISOLATE_EVENTS = True
    DEFAULT_EVENT_THRESHOLD = 0.5
    # DEFAULT_EVALUATION = "time"

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
        # start = x.index[0].timestamp()
        end = start + step / 1000
        # interval = pd.Interval(start, end)
        # tmp = tags.id[tags.index.overlaps(interval)].unique()
        tmp = np.unique(tags["id"][(tags["start"] < end) & (tags["end"] >= start)])
        idx = ",".join(tmp)
        # idx = ",".join(list(map(str, tmp)))
        return idx

    def isolate_events(self, predictions, step):
        tmp = predictions.loc[predictions.event > 0]
        tmp.reset_index(inplace=True)

        step = datetime.timedelta(milliseconds=step)
        start = None
        events = []
        event_id = 1
        if not tmp.empty:
            for _, x in tmp.iterrows():
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
                    "recording_id": x.recording_id,
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
                predictions.tag.iloc[start:end] = 1
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
        res["recording_id"] = recording_id
        res.loc[res.tag_index != "", "tag"] = 1
        return res

    def get_stats(self, df, expand_index=False):
        # TODO: make stats coherent between no resampling and resampling: use tag index in the same way
        res = df.tag + df.event

        n_true_positives = len(res[res == 3])
        n_true_negatives = len(res[res == 0])
        n_false_positives = len(res[res == 2])
        n_false_negatives = len(res[res == 1])

        if expand_index:
            df2 = df.loc[(df.tag_index != "") & (df.event > 0)]
            tmp = df2.tag_index.unique()
            matched_tags = set(",".join(tmp).split(","))
            all_tags = df.tag_index[df.tag_index != ""].unique()
            all_tags = set(",".join(all_tags).split(","))
        else:
            df2 = df.loc[(df.tag_index > -1) & (df.event > 0)]
            matched_tags = df2.tag_index.unique()
            all_tags = df.tag_index[df.tag_index > -1].unique()

        fn2 = len(all_tags) - len(matched_tags)

        recall2 = n_true_positives / (n_true_positives + fn2)
        precision = round(n_true_positives / (n_true_positives + n_false_positives), 3)
        recall = round(n_true_positives / (n_true_positives + n_false_negatives), 3)
        f1_score = round(2 * precision * recall / (precision + recall), 3)
        return {
            "n_true_positives": n_true_positives,
            "n_false_positives": n_false_positives,
            "n_true_negatives": n_true_negatives,
            "n_false_negatives": n_false_negatives,
            "precision": precision,
            "recall": recall,
            "recall2": recall2,
            "f1_score": f1_score,
        }

    def evaluate(self, predictions, tags, options):
        preds = predictions.copy()
        preds.loc[:, "tag"] = 0
        preds.loc[:, "tag_index"] = -1
        preds.loc[:, "event"] = 0
        preds.loc[:, "datetime"] = pd.to_datetime(preds.time * 10 ** 9)
        preds.set_index("datetime", inplace=True)
        step = options.get("sample_step", self.DEFAULT_MIN_DURATION) * 1000

        if step:
            apply_func = self.match_events_apply
            as_index = False
            expand_index = True

        else:
            apply_func = self.get_recording_events
            as_index = True
            expand_index = False

        events = preds.groupby("recording_id", as_index=as_index, observed=True).apply(
            apply_func, tags, options
        )
        print(events)
        stats = self.get_stats(events, expand_index=expand_index)
        print("Stats for options {0}: {1}".format(options, stats))
        return {"options": options, "stats": stats, "events": events}

    # def evaluate_by_time(self, predictions, tags, options):
    #     preds = predictions.copy()
    #     tags = tags.copy()
    #     preds.loc[:, "tag"] = -1
    #     preds.loc[:, "tag_index"] = -1
    #     preds.loc[:, "event"] = -1
    #     preds.loc[:, "datetime"] = pd.to_datetime(preds.time * 10**9)
    #     preds.set_index("datetime", inplace=True)

    #     events = preds.groupby("recording_id", as_index=False).apply(
    #         self.match_events, tags, options)
    #     events["tag"] = 0
    #     events.loc[events.tag_index != "", "tag"] = 1
    #     stats = self.get_stats(events, expand_index=True)
    #     print("Stats for options {0}: {1}".format(options, stats))
    #     return [options, stats, events]

    # def evaluate_by_events(self, predictions, tags, options):
    #     pass
