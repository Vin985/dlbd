import pandas as pd
from mouffet.evaluation.detector import Detector
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_bar,
    geom_text,
    ggplot,
    ggtitle,
    theme,
    theme_classic,
    xlab,
    ylab,
)
from plotnine.positions.position_dodge import position_dodge


class StandardDetector(Detector):

    REQUIRES = ["tags_df"]

    DEFAULT_MIN_DURATION = 0.1
    DEFAULT_END_THRESHOLD = 0.6

    def get_recording_events(self, predictions, options=None):
        options = options or {}
        min_activity = options.get(
            "activity_threshold", self.DEFAULT_ACTIVITY_THRESHOLD
        )
        end_threshold = options.get("end_threshold", self.DEFAULT_END_THRESHOLD)
        min_duration = options.get("min_duration", self.DEFAULT_MIN_DURATION)
        event_index = 0
        ongoing = False
        events = []
        start = 0
        end = 0

        for activity, recording_id, pred_time in predictions.itertuples(index=False):
            # * Check if prediction is above a defined threshold
            if activity > min_activity:
                # * If not in a song, create a new event
                if not ongoing:
                    ongoing = True
                    event_index += 1
                    start = pred_time
            elif ongoing:
                # * if above an end threshold, consider it as a single event
                if activity > end_threshold:
                    continue
                # * If below the threshold and in an active event, end it
                ongoing = False
                end = pred_time
                # * log event if its duration is greater than minimum threshold
                if (end - start) > min_duration:
                    events.append(
                        {
                            "event_index": event_index,
                            "recording_id": recording_id,
                            "start": start,
                            "end": end,
                        }
                    )
        if ongoing:
            end = pred_time
            if end - start > min_duration:
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

    def get_events(self, predictions, options, *args, **kwargs):
        predictions = predictions[["activity", "recording_id", "time"]]
        events = predictions.groupby("recording_id", as_index=False, observed=True)
        events = events.apply(self.get_recording_events, options)
        events.reset_index(inplace=True)
        events.drop(["level_0", "level_1"], axis=1, inplace=True)
        events["event_duration"] = events["end"] - events["start"]
        events.reset_index(inplace=True)
        events = events[self.EVENTS_COLUMNS.keys()]
        events.rename(columns=self.EVENTS_COLUMNS, inplace=True)
        return events

    def get_matches(self, events, tags, options):
        tags = tags.rename(columns=self.TAGS_COLUMNS_RENAME)
        tmp = tags.merge(events, on="recording_id", how="outer")
        # * Select tags associated with an event
        matched = tmp.loc[
            (tmp.event_start <= tmp.tag_end) & (tmp.event_end >= tmp.tag_start)
        ]
        # * Get list of tags not associated with an event
        unmatched = tags.loc[~tags.tag_id.isin(matched.tag_id.unique())]
        # * Add them to the final dataframe
        match_df = matched.merge(unmatched, how="outer")
        match_df.loc[match_df.event_id.isna(), "event_id"] = -1
        match_df.event_id = match_df.event_id.astype("int")
        match_df.reset_index(inplace=True)
        match_df = self.get_overlap_duration(match_df, "event")
        match_df = self.get_overlap_duration(match_df, "tag")

        events.loc[:, "matched"] = 0
        tags.loc[:, "matched"] = 0

        if not match_df.empty:
            dtc_threshold = options.get("dtc_threshold", 0.3)
            gtc_threshold = options.get("gtc_threshold", 0.1)
            res = match_df.loc[
                (match_df.event_overlap >= dtc_threshold)
                & (match_df.tag_overlap >= gtc_threshold)
            ]

            true_positives_id = res.event_id.unique()
            matched_tags_id = res.tag_id.unique()

            events.loc[events.event_id.isin(true_positives_id), "matched"] = 1
            tags.loc[tags.tag_id.isin(matched_tags_id), "matched"] = 1

        return events, tags, match_df

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

    def get_proportions(self, x, by, col="", total=0):
        if not total:
            total = x.shape[0]
        if by:
            by = by.copy()
            ncol = by.pop(0)
            res = x.groupby(ncol).apply(self.get_proportions, by, ncol, x.shape[0])
        else:
            res = x.iloc[0]
            res["n_tags"] = x.shape[0]
        if col:
            res["prop_" + col] = x.shape[0] / total * 100
            res["lbl_" + col] = "n = {}\n({}%)\n\n".format(
                x.shape[0], round(x.shape[0] / total * 100, 2)
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

    def get_tag_repartition(self, tag_df, options):
        if not "background" in tag_df.columns:
            tag_df["background"] = False
        test = tag_df[["tag", "matched", "background", "id"]].copy()
        test.loc[:, "prop_matched"] = -1
        test.loc[:, "prop_background"] = -1
        test.loc[:, "lbl_matched"] = ""
        test.loc[:, "lbl_background"] = ""
        test.loc[:, "n_tags"] = -1

        n_total = test.shape[0]
        n_matched = test.matched.value_counts()

        tags_summary = (
            test.groupby("tag")
            .apply(self.get_proportions, by=["matched", "background"])
            .reset_index(drop=True)
        )
        tags_summary = tags_summary.sort_values(["tag", "matched", "background"])

        plt = ggplot(
            data=tags_summary,
            mapping=aes(
                x="tag",  # "factor(species, ordered=False)",
                y="n_tags",
                fill="background",
                ymax=max(tags_summary.n_tags) + 35,  # "factor(species, ordered=False)",
            ),
        )
        plot_width = 10 + len(tags_summary.tag.unique()) * 0.75
        plt = (
            plt
            + geom_bar(stat="identity", show_legend=True, position=position_dodge())
            + facet_wrap(
                "matched",
                nrow=1,
                ncol=2,
                scales="fixed",
                labeller=(lambda x: self.get_matched_label(x, n_total, n_matched)),
            )
            + xlab("Species")
            + ylab("Number of annotations")
            + geom_text(
                mapping=aes(label="lbl_background"), position=position_dodge(width=0.9),
            )
            + geom_text(
                mapping=aes(y=max(tags_summary.n_tags) + 30, label="lbl_matched",)
            )
            + theme_classic()
            + theme(
                axis_text_x=element_text(angle=90, vjust=1, hjust=1, margin={"r": -30}),
                plot_title=element_text(
                    weight="bold", size=14, margin={"t": 10, "b": 10}
                ),
                figure_size=(plot_width, 10),
                text=element_text(size=12, weight="bold"),
            )
            + ggtitle(
                (
                    "Tag repartition for model {}, database {}, class {}\n"
                    + "with detector options {}"
                ).format(
                    options["scenario_info"]["model"],
                    options["scenario_info"]["database"],
                    options["scenario_info"]["class"],
                    options,
                )
            )
        )

        return plt

    def get_stats(self, events, tags, options):

        n_events = events.shape[0]
        n_tags = tags.shape[0]

        n_true_positives = events.matched.sum()
        n_false_positives = n_events - n_true_positives

        n_tags_matched = tags.matched.sum()
        n_tags_unmatched = n_tags - n_tags_matched

        precision = round(n_true_positives / n_events, 3)
        recall = round(n_tags_matched / n_tags, 3) if n_tags else 0

        if precision + recall:
            f1_score = round(2 * precision * recall / (precision + recall), 3)
        else:
            f1_score = 0

        true_positives_ratio = round(n_true_positives / n_tags, 3) if n_tags else 0

        tags_active_dur = self.tags_active_duration(tags)
        false_positive_rate = (
            round(n_false_positives / tags_active_dur, 3) if tags_active_dur else 0
        )

        stats = pd.DataFrame(
            [
                {
                    "n_events": events.shape[0],
                    "n_tags": tags.shape[0],
                    "n_true_positives": n_true_positives,
                    "n_false_positives": n_false_positives,
                    "n_matched_tags": n_tags_matched,
                    "n_unmatched_tags": n_tags_unmatched,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "true_positives_ratio": true_positives_ratio,
                    "false_positive_rate": false_positive_rate,
                }
            ]
        )

        return stats

    def draw_plots(self, tags, options):
        res = {}
        if options.get("plot_tag_repartition", True):
            tag_repartition = self.get_tag_repartition(tags, options)
            res["tag_repartition"] = tag_repartition
        return res

    def evaluate(self, predictions, tags, options):
        tags = tags["tags_df"]
        events = self.get_events(predictions, options)
        events, tags, matches = self.get_matches(events, tags, options)
        stats = self.get_stats(events, tags, options)

        res = {
            "stats": stats,
            "matches": matches,
        }
        if options.get("draw_plots", True):
            res["plots"] = self.draw_plots(tags=tags, options=options)
        print("Stats for options {0}:\n {1}".format(options, stats))
        return res

    # def plot_PR_curve(self, stats, options):
    #     return stats
