import datetime
import functools

import numpy as np
import pandas as pd

from .detector import Detector


class SubsamplingDetector(Detector):

    DEFAULT_ISOLATE_EVENTS = True
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

    def get_recording_events(self, predictions, options=None):
        recording_id = predictions.recording_id.values[0]
        preds = predictions[["time", "activity"]].copy()
        min_activity = options.get("min_activity", self.DEFAULT_MIN_ACTIVITY)
        step = options.get("min_duration", self.DEFAULT_MIN_DURATION) * 1000
        isolate_events = options.get("isolate_events", self.DEFAULT_ISOLATE_EVENTS)

        preds.loc[:, "datetime"] = pd.to_datetime(preds.time * 10 ** 9)
        preds.set_index("datetime", inplace=True)

        resampled = preds.resample(str(step) + "ms")
        # TODO: add resampling method in options
        resample_func = functools.partial(self.resample_max, threshold=min_activity)
        res = resampled.agg({"activity": resample_func})
        res.rename(columns={"activity": "event"}, inplace=True)
        res["recording_id"] = recording_id

        if isolate_events:
            return self.isolate_events(res, step)

        return res

    def match_events_apply(self, predictions, tags, options):
        recording_id = predictions.name
        current_tags = tags.loc[tags.recording_id == recording_id]
        min_activity = options.get("min_activity", self.DEFAULT_MIN_ACTIVITY)
        step = options.get("min_duration", self.DEFAULT_MIN_DURATION) * 1000

        resampled = predictions.resample(str(step) + "ms")
        resample_func = functools.partial(self.resample_max, threshold=min_activity)
        tmp_tags = {
            "id": current_tags.id.to_numpy().astype(str),
            "start": current_tags.tag_start.to_numpy(),
            "end": current_tags.tag_end.to_numpy(),
        }
        tag_func = functools.partial(self.get_tag_index, step=step, tags=tmp_tags)
        res = resampled.agg({"activity": resample_func, "tag_index": tag_func})
        res["recording_id"] = recording_id
        res.rename(columns={"activity": "event"}, inplace=True)
        return res

    def get_stats(self, df, expand_index=False):
        if not "res" in df.columns:
            df.loc[:, "res"] = df.tag + df.event

        tp = len(df.res[df.res == 3])
        tn = len(df.res[df.res == 0])
        fp = len(df.res[df.res == 2])
        fn = len(df.res[df.res == 1])

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

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        recall2 = tp / (tp + fn2)
        return {
            "precision": precision,
            "recall": recall,
            "recall2": recall2,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "fn2": fn2,
        }

    def get_events(self, predictions, options=None):
        options = options or {}
        events = predictions.groupby("recording_id", as_index=False, observed=True)

        if options.get("parallel", True):
            events = events.parallel_apply(self.get_recording_events, options)
        else:
            events = events.apply(self.get_recording_events, options)
        events.reset_index(inplace=True)
        events.drop(["level_0"], axis=1, inplace=True)
        return events

    def evaluate(self, predictions, tags, options):
        preds = predictions.copy()
        preds.loc[:, "tag"] = -1
        preds.loc[:, "tag_index"] = -1
        preds.loc[:, "event"] = -1
        preds.loc[:, "datetime"] = pd.to_datetime(preds.time * 10 ** 9)
        preds.set_index("datetime", inplace=True)

        events = preds.groupby("recording_id", as_index=False).apply(
            self.match_events_apply, tags, options
        )
        events["tag"] = 0
        events.loc[events.tag_index != "", "tag"] = 1
        stats = self.get_stats(events, expand_index=True)
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
