import pandas as pd
import numpy as np

from .detector import Detector


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
        }
        return {
            "stats": stats,
            "events": events,
            "tags": tags,
            "matched": matched,
            "options": options,
        }

    def evaluate(self, predictions, tags, options):
        events = self.get_events(predictions, options)
        matches = self.get_matches(events, tags)
        stats = self.get_stats(events, tags, matches, options)
        print("Stats for options {0}: {1}".format(options, stats["stats"]))
        return stats
