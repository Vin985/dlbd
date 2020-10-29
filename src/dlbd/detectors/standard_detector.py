import pandas as pd
import numpy as np

from .detector import Detector

from plotnine import (
    aes,
    element_text,
    geom_bar,
    geom_text,
    ggplot,
    ggtitle,
    save_as_pdf_pages,
    scale_x_discrete,
    facet_wrap,
    theme,
    theme_classic,
    xlab,
    ylab,
)
from plotnine.labels import ggtitle
from plotnine.positions.position_dodge import position_dodge

from functools import partial


class StandardDetector(Detector):
    def __init__(self):
        super().__init__()

    def get_events(self, predictions, options):
        predictions = predictions[["activity", "recording_id", "time"]]
        events = predictions.groupby("recording_id", as_index=False, observed=True)
        # if options.get("parallel", True):
        #     print("paralleling")
        #     events = events.parallel_apply(self.get_recording_events, options)
        # else:
        events = events.apply(self.get_recording_events, options)
        events.reset_index(inplace=True)
        events.drop(["level_0", "level_1"], axis=1, inplace=True)
        events["event_duration"] = events["end"] - events["start"]
        events.reset_index(inplace=True)
        events = events[self.EVENTS_COLUMNS.keys()]
        events.rename(columns=self.EVENTS_COLUMNS, inplace=True)
        return events

    def get_recording_events(self, predictions, options=None):
        options = options or {}
        min_activity = options.get("min_activity", self.DEFAULT_MIN_ACTIVITY)
        end_threshold = options.get("end_threshold", self.DEFAULT_END_THRESHOLD)
        min_duration = options.get("min_duration", self.DEFAULT_MIN_DURATION)
        event_index = 0
        ongoing = False
        events = []
        start = 0
        end = 0
        # def detect_songs_events(predictions):
        for activity, recording_id, pred_time in predictions.itertuples(index=False):
            # Check if prediction is above a defined threshold
            if activity > min_activity:
                # If not in a song, create a new event
                if not ongoing:
                    ongoing = True
                    event_index += 1
                    start = pred_time
            elif ongoing:
                # if above an end threshold, consider it as a single event
                if activity > end_threshold:
                    continue
                # If below the threshold and in an active event, end it
                ongoing = False
                end = pred_time
                # log event if its duration is greater than minimum threshold
                if (end - start) > min_duration:
                    events.append(
                        {
                            "event_index": event_index,
                            "recording_id": recording_id,
                            "start": start,
                            "end": end,
                        }
                    )
        events = pd.DataFrame(events)
        return events

    def associate_recordings(self, events, recordings):
        events = events.merge(
            recordings[["id", "name"]], left_on="recording_id", right_on="id"
        )
        events.reset_index(inplace=True)
        events = events[self.EVENTS_COLUMNS.keys()]
        events.rename(columns=self.EVENTS_COLUMNS, inplace=True)
        return events

    def get_matches(self, events, tags):
        tags = tags.rename(columns=self.TAGS_COLUMNS_RENAME)
        tmp = tags.merge(events, on="recording_id", how="outer")
        # Select tags associated with an event
        matched = tmp.loc[
            (tmp.event_start <= tmp.tag_end) & (tmp.event_end >= tmp.tag_start)
        ]
        # Get list of tags not associated with an event
        unmatched = tags.loc[~tags.tag_id.isin(matched.tag_id.unique())]
        # add them to the final dataframe
        match_df = matched.merge(unmatched, how="outer")
        match_df.loc[match_df.event_id.isna(), "event_id"] = -1
        match_df.event_id = match_df.event_id.astype("int")
        match_df.reset_index(inplace=True)

        return match_df

    def match_predictions(self, predictions, events, tags):
        preds = predictions.copy()
        preds["tag"] = 0
        preds["event"] = 0
        preds["event_id"] = -1
        preds["tag_id"] = -1
        for _, x in tags.iterrows():
            preds.loc[
                preds.time.between(x["tag_start"], x["tag_end"]), ["tag", "tag_id"]
            ] = [1, x["id"]]

        for _, x in events.iterrows():
            preds.loc[
                preds.time.between(x["event_start"], x["event_end"]),
                ["event", "event_id"],
            ] = [2, x["event_id"]]
        return preds

    def get_stats_old(self, events, matches):
        # True pos: number of unique events that matched with a tag
        true_pos_events = len(matches[matches.event_id != -1].event_id.unique())
        true_pos_tags = len(matches[matches.event_id != -1].tag_id.unique())
        # False neg: number of tags that did not have a match
        false_neg = matches[matches.event_id == -1].shape[0]
        # Number of tags that are matched
        n_tags_matched = len(matches.loc[matches.event_id != -1].tag_id.unique())

        # Precision: TP / TP + FP
        precision = true_pos_events / events.shape[0]
        # Recall: TP / TP + FN
        recall = true_pos_tags / (true_pos_tags + false_neg)
        f1_score = 2 * (precision * recall) / (precision + recall)

        return {
            "n_events": events.shape[0],
            "n_tags": len(matches.tag_id.unique()),
            "true_positive_events": true_pos_events,
            "true_positive_tags": true_pos_tags,
            "false_negative": false_neg,
            "n_tags_matched": n_tags_matched,
            "precision": precision,
            "recall": recall,
            "F1_score": f1_score,
        }

    @staticmethod
    def event_overlap_duration(tags):
        tags.sort_values("tag_start")
        previous_end = tags.iloc[0].event_start
        overlap_duration = 0
        for tag in tags.itertuples():
            if tag.tag_end > previous_end:
                end = min(tag.tag_end, tag.event_end)
                start = max(tag.tag_start, previous_end)
                overlap_duration += end - start
                if end == tag.event_end:
                    break
                previous_end = end
        return overlap_duration

    @staticmethod
    def tag_overlap_duration(events):
        overlap_duration = 0
        for event in events.itertuples():
            overlap_duration += min(event.tag_end, event.event_end) - max(
                event.tag_start, event.event_start
            )
        return overlap_duration

    def get_overlap_duration(self, match_df, overlap_type):
        overlap_func = getattr(self, overlap_type + "_overlap_duration")
        if not match_df.empty:
            overlap_duration = (
                match_df.groupby(overlap_type + "_id")
                .apply(overlap_func)
                .rename(overlap_type + "_overlap_duration")
                .reset_index()
            )
            tmp = match_df.merge(overlap_duration)
            tmp[overlap_type + "_overlap"] = (
                tmp[overlap_type + "_overlap_duration"]
                / tmp[overlap_type + "_duration"]
            )
            return tmp
        return pd.DataFrame()

    @staticmethod
    def tags_active_duration(tags):
        duration = 0
        previous_start, previous_end = 0, 0
        for tag in tags.itertuples():
            start = tag.tag_start
            end = tag.tag_end
            if previous_start < start < previous_end:
                if end > previous_end:
                    duration += end - previous_end
                    previous_end = end
            else:
                previous_start = start
                previous_end = end
                duration += end - start
        return duration

    @staticmethod
    def get_proportion(x, n_total=-1):
        print(x)
        if n_total < 0:
            n_total = x.shape[0]
            # raise ValueError("n_total should be provided and positive")
        return round(len(x) / n_total * 100, 1)

    def test_func(self, x, n_total):
        total_func = partial(self.get_proportion, n_total=n_total)
        match_func = partial(self.get_proportion, n_total=x.shape[0])
        res = (
            x.groupby(["tag", "background"])
            .agg({"tag": "count", "prop_total": total_func, "prop_match": match_func})
            .rename(columns={"tag": "n_tags"})
            .reset_index()
            .astype({"background": "category", "tag": "category"})
        )
        return res

    @staticmethod
    def get_matched_label(x, n_total, n_matched):
        x = int(x)
        label = x == 1 and "matched" or "unmatched"
        label += " (n={}, {}%)".format(
            n_matched[x], round(n_matched[x] / n_total * 100, 1)
        )
        return label

    def get_tag_repartition(self, tag_df):
        test = tag_df[["tag", "matched", "background", "id"]].copy()
        test.loc[:, "prop_total"] = -1
        test.loc[:, "prop_match"] = -1
        test.loc[:, "tag_match"] = -1

        n_total = test.shape[0]
        n_matched = test.matched.value_counts()

        test2 = test.groupby("matched").apply(self.test_func, n_total=test.shape[0])
        print(test2.reset_index())
        print(test2.dtypes)
        if "background" in tag_df.columns:
            tags_summary = (
                tag_df.groupby(["matched", "tag", "background"])
                .agg({"tag": "count"})
                .rename(columns={"tag": "n_tags"})
                .reset_index()
                .astype(
                    {"background": "category", "tag": "category", "matched": "category"}
                )
            )
            print(tags_summary)
            plt = ggplot(
                data=tags_summary,
                mapping=aes(
                    x="tag",  # "factor(species, ordered=False)",
                    y="n_tags",
                    fill="background",  # "factor(species, ordered=False)",
                ),
            )
        else:
            tags_summary = (
                tag_df.groupby(["tag", "matched"])
                .agg({"tag": "count"})
                .rename(columns={"tag": "n_tags"})
                .reset_index()
                .astype({"tag": "category", "matched": "category"})
            )
            plt = ggplot(
                data=tags_summary,
                mapping=aes(x="tag", y="n_tags",),  # "factor(species, ordered=False)",
            )

        plt = (
            plt
            + geom_bar(stat="identity", show_legend=True, position=position_dodge())
            + facet_wrap(
                "matched",
                nrow=1,
                ncol=2,
                scales="fixed",
                labeller=(lambda x: self.get_matched_label(x, n_total, n_matched)),
                # partial(
                #     self.get_matched_label, n_total=n_total, n_matched=n_matched
                # )
                # (lambda x: x == "1" and "matched" or "unmatched"),
            )
            + xlab("Species")
            + ylab("Number of annotations")
            + geom_text(
                mapping=aes(y=tags_summary.n_tags + 2, label="n_tags"),
                position=position_dodge(width=0.9),
            )
            + geom_text(
                mapping=aes(
                    x=[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                    y=200,
                    label=[
                        "toto",
                        "tutu",
                        "titi",
                        "",
                        "",
                        "",
                        "toto2",
                        "titi2",
                        "tutu2",
                        "",
                        "",
                        "",
                    ],
                )
            )
            + theme_classic()
            + theme(
                axis_text_x=element_text(angle=90, vjust=1, hjust=1, margin={"r": -30}),
                figure_size=(20, 8),
            )
            # + ggtitle(
            #     "_".join([database["name"], db_type, "tag_species.png"])
            #     + "(n = "
            #     + str(tag_df.shape[0])
            #     + ")"
            # )
            # + scale_x_discrete(limits=SPECIES_LIST, labels=xlabels)
        )

        return plt

    def get_stats(self, events, tags, matches, options):

        matched = matches.loc[
            matches.event_id != -1,
            [
                "event_id",
                "tag_id",
                "tag_start",
                "tag_end",
                "tag_duration",
                "event_start",
                "event_end",
                "event_duration",
            ],
        ].copy()

        if matched.empty:
            res = matched
        else:
            matched = self.get_overlap_duration(matched, "event")
            matched = self.get_overlap_duration(matched, "tag")

            dtc_threshold = options.get("dtc_threshold", 0.3)
            gtc_threshold = options.get("gtc_threshold", 0.1)
            res = matched.loc[
                (matched.event_overlap >= dtc_threshold)
                & (matched.tag_overlap >= gtc_threshold)
            ]

        true_positives = res.event_id.unique()
        n_true_positives = len(true_positives)
        n_false_positives = events.shape[0] - n_true_positives
        true_positives_ratio = round(n_true_positives / tags.shape[0], 3)
        false_positive_rate = round(
            n_false_positives / self.tags_active_duration(tags), 3
        )
        precision = round(n_true_positives / events.shape[0], 3)

        matched_tags_id = res.tag_id.unique()
        tags.loc[:, "matched"] = 0
        tags.loc[tags.id.isin(matched_tags_id), "matched"] = 1
        n_tags_matched = len(matched_tags_id)
        n_tags_unmatched = tags.shape[0] - n_tags_matched
        recall = round(n_tags_matched / tags.shape[0], 3)

        events.loc[:, "matched"] = 0
        events.loc[events.event_id.isin(true_positives), "matched"] = 1

        f1_score = round(2 * precision * recall / (precision + recall), 3)

        stats = {
            "n_events": events.shape[0],
            "n_tags": tags.shape[0],
            "n_true_positives": n_true_positives,
            "n_false_positives": n_false_positives,
            "n_tags_matched": n_tags_matched,
            "n_tags_unmatched": n_tags_unmatched,
            "true_positives_ratio": true_positives_ratio,
            "false_positive_rate": false_positive_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

        tag_repartition = self.get_tag_repartition(tags)
        return stats, tag_repartition

    def evaluate(self, predictions, tags, options):
        events = self.get_events(predictions, options)
        matches = self.get_matches(events, tags)
        stats, tag_repartition = self.get_stats(events, tags, matches, options)
        print("Stats for options {0}: {1}".format(options, stats))
        return {
            "options": options,
            "stats": stats,
            "matches": matches,
            "tag_repartition": tag_repartition,
        }

