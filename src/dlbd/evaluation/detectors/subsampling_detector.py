import datetime
import functools
import math
from functools import partial

import numpy as np
import pandas as pd
import plotnine

from mouffet.evaluation.detector import Detector

from plotnine import (
    aes,
    ggplot,
    geom_line,
)


class SubsamplingDetector(Detector):

    REQUIRES = ["tags_df"]

    DEFAULT_ISOLATE_EVENTS = True
    DEFAULT_EVENT_THRESHOLD = 0.5

    def resample_max(self, x, threshold=0.98):
        if max(x) >= threshold:
            return 1
        return 0

    def has_tag(self, x):
        if max(x) > 0:
            return 2
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
        predictions.loc[predictions.activity > threshold, "event"] = 1
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
        threshold = options.get("event_threshold", self.DEFAULT_EVENT_THRESHOLD)
        step = options.get("sample_step", self.DEFAULT_MIN_DURATION) * 1000
        resampler = predictions.resample(str(step) + "ms") if step else predictions
        resample_func = functools.partial(self.resample_max, threshold=threshold)
        tmp_tags = {
            "id": current_tags.id.to_numpy().astype(str),
            "start": current_tags.tag_start.to_numpy(),
            "end": current_tags.tag_end.to_numpy(),
        }
        tag_func = functools.partial(self.get_tag_index, step=step, tags=tmp_tags)
        res = (
            resampler.agg({"activity": resample_func, "tag_index": tag_func})
            .reset_index()
            .rename(columns={"activity": "event", "datetime": "time"})
        )
        # res["recording_id"] = recording_id
        return res

    def recording_plot(self, matches, options, tags_presence, step):

        # y_true = y_true > 0.5

        plot = (
            ggplot(
                data=matches,
                mapping=aes(x="time", y="tag",),  # "factor(species, ordered=False)",
            )
            + geom_line(mapping=aes(y="event"), color="pink")
            + geom_line(color="blue")
            + geom_line(
                mapping=aes(
                    y=options.get("event_threshold", self.DEFAULT_EVENT_THRESHOLD)
                ),
                color="red",
            )
        )
        if not step:
            hww = 10
            tags = matches.tag.values
            y_true = np.zeros(len(tags))
            for idx, _ in enumerate(tags):
                start = max(idx - hww, 0)
                end = min(idx + hww, tags.shape[0])
                r = 1 if tags[start:end].max() > 0.5 else 0
                y_true[idx] = r
            plot = (
                plot
                + geom_line(mapping=aes(y=y_true), color="purple")
                + geom_line(mapping=aes(y="activity"), color="orange")
            )

        return plot

    def get_recording_plots(self, matches, options, tags_presence, step):
        plots = matches.groupby("recording_id").apply(
            self.recording_plot, options, tags_presence, step
        )
        plotnine.save_as_pdf_pages(plots, "test_save_plots_citynet.pdf")

    def get_stats(self, df, tags):
        # TODO: make stats coherent between no resampling and resampling: use tag index in the same way
        res = df.tag + df.event

        n_true_positives = len(res[res == 3])
        n_true_negatives = len(res[res == 0])
        n_false_positives = len(res[res == 1])
        n_false_negatives = len(res[res == 2])

        tagged_samples = len(df.tag[df.tag == 2])

        matched_tags = set(
            df.tag_index.loc[(df.tag_index.str.len() > 0) & (df.event > 0)].sum()
        )

        n_matched_tags = len(matched_tags)

        n_tags = tags.shape[0]
        n_unmatched_tags = n_tags - n_matched_tags

        precision = round(n_true_positives / (n_true_positives + n_false_positives), 3)
        recall_samples = round(
            n_true_positives / (n_true_positives + n_false_negatives), 3
        )
        recall_tags = round(n_matched_tags / (n_matched_tags + n_unmatched_tags), 3)
        f1_score = round(
            2 * precision * recall_samples / (precision + recall_samples), 3
        )
        return {
            "n_events": int(df.event.sum()),
            "n_tags": n_tags,
            "n_true_positives": n_true_positives,
            "n_false_positives": n_false_positives,
            "n_matched_tags": n_matched_tags,
            "n_unmatched_tags": n_unmatched_tags,
            "n_tagged_samples": tagged_samples,
            "precision": precision,
            "recall_sample": recall_samples,
            "recall_tags": recall_tags,
            "f1_score": f1_score,
        }

    def evaluate(self, predictions, tags, options):
        tags_df = tags["tags_df"]
        # tags_presence = {}
        # for idx, i in enumerate(tags["tags_linear_presence"]):
        #     fp = str(tags["infos"][idx]["file_path"])
        #     tags_presence[fp] = i

        preds = predictions.copy()
        # preds.loc[:, "tag_index"] = ""
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
            preds.groupby("recording_id", observed=True)
            .apply(apply_func, tags=tags_df, options=options)
            .reset_index()
        )
        events.loc[:, "tag"] = 0
        events.loc[events.tag_index.str.len() > 0, "tag"] = 2

        stats = self.get_stats(events, tags_df)
        print("Stats for options {0}: {1}".format(options, stats))
        # self.get_recording_plots(events, options, tags_presence, step)
        return {"options": options, "stats": stats, "matches": events}