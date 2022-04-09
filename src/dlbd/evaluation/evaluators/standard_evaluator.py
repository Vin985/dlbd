import pandas as pd
from mouffet import common_utils
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_bar,
    geom_point,
    geom_text,
    ggplot,
    ggtitle,
    theme,
    theme_classic,
    xlab,
    ylab,
)
from plotnine.positions.position_dodge import position_dodge

from . import SongDetectorEvaluator
from ...data.tag_utils import flatten_tags


class StandardEvaluator(SongDetectorEvaluator):

    REQUIRES = ["tags_df"]

    DEFAULT_MIN_DURATION = 0.1
    DEFAULT_END_THRESHOLD = 0.6

    DEFAULT_PLOTS = ["detected_tags", "overlap_duration"]

    MATCH_TYPES = {
        "recording_id": "category",
        "tag": "category",
        "noise": "int32",
        "background": "bool",
        "file_name": "category",
        "tag_id": "int32",
        "tag_index": "int32",
    }

    def file_event_duration(self, df):
        # * Returns total duration of events in a file
        if df.empty:
            return 0
        return df.drop_duplicates("event_id")["event_duration"].sum()

    def file_tag_duration(self, df):
        flattened = flatten_tags(df.drop_duplicates("tag_id"))
        return flattened.tag_duration.sum()

    def get_recording_events(self, predictions, options=None):
        # print("events for recording {}".format(predictions.name))
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
        buffer = options.get("buffer", 0)
        last_time = predictions.iloc[predictions.shape[0] - 1].time

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
                            # "event_index": event_index,
                            "recording_id": recording_id,
                            "start": max(start - buffer, 0),
                            "end": min(end + buffer, last_time),
                        }
                    )
        if ongoing:
            end = pred_time
            if end - start > min_duration:
                events.append(
                    {
                        # "event_index": event_index,
                        "recording_id": recording_id,
                        "start": max(start - buffer, 0),
                        "end": end,
                    }
                )
        if events:
            events = pd.DataFrame(events)
        else:
            events = pd.DataFrame(columns=["recording_id", "start", "end"])
        return events

    def filter_predictions(self, predictions, options, *args, **kwargs):
        predictions = predictions[["activity", "recording_id", "time"]]
        events = predictions.groupby("recording_id", as_index=False, observed=True)
        events = (
            events.apply(self.get_recording_events, options)
            .reset_index()
            .drop(["level_0", "level_1"], axis=1)
        )
        events["event_duration"] = events["end"] - events["start"]
        events.reset_index(inplace=True)
        events = events[self.EVENTS_COLUMNS.keys()]
        events.rename(columns=self.EVENTS_COLUMNS, inplace=True)
        events.recording_id = events.recording_id.astype("category")
        return events

    @staticmethod
    def event_overlap_duration(tags):
        if tags.name == -1:
            return pd.DataFrame([{"event_overlap_duration": 0, "iou": 0}])
        tmp = tags.sort_values("tag_start")
        previous_end = tmp.iloc[0].event_start
        overlap_duration = 0
        for tag in tmp.itertuples():
            if tag.tag_end > previous_end:
                end = min(tag.tag_end, tag.event_end)
                start = max(tag.tag_start, previous_end)
                overlap_duration += end - start
                if end == tag.event_end:
                    break
                previous_end = end
        union = max(tags.tag_end.max(), tags.event_end.max()) - min(
            tags.tag_start.min(), tags.event_start.min()
        )
        return pd.DataFrame(
            [
                {
                    "event_overlap_duration": overlap_duration,
                    "iou": overlap_duration / union,
                }
            ]
        )

    @staticmethod
    def tag_overlap_duration(events):
        overlap_duration = 0
        for event in events.itertuples():
            overlap_duration += min(event.tag_end, event.event_end) - max(
                event.tag_start, event.event_start
            )
        if overlap_duration > event.tag_duration:
            overlap_duration = event.tag_duration
        return pd.DataFrame([{"tag_overlap_duration": overlap_duration}])

    def get_overlap_duration(self, match_df, overlap_type):
        overlap_func = getattr(self, overlap_type + "_overlap_duration")
        if not match_df.empty:
            tmp_df = match_df[
                [
                    "event_id",
                    "tag_id",
                    "tag_start",
                    "tag_end",
                    "event_start",
                    "event_end",
                    "tag_duration",
                    "event_duration",
                ]
            ]
            overlap_duration = (
                tmp_df.groupby(overlap_type + "_id").apply(overlap_func)
                # .rename(overlap_type + "_overlap_duration")
            )
            overlap_duration = overlap_duration.reset_index()
            tmp = match_df.merge(overlap_duration)
            tmp[overlap_type + "_overlap"] = (
                tmp[overlap_type + "_overlap_duration"]
                / tmp[overlap_type + "_duration"]
            )
            return tmp
        return pd.DataFrame()

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
        types = {k: v for k, v in self.MATCH_TYPES.items() if k in match_df.columns}
        match_df = match_df.astype(types)
        match_df = self.get_overlap_duration(match_df, "event")
        match_df = self.get_overlap_duration(match_df, "tag")

        if not events.empty:
            events.loc[:, "matched"] = 0
        else:
            events = events.assign(matched=None)
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

            if not events.empty:
                events.loc[events.event_id.isin(true_positives_id), "matched"] = 1
            tags.loc[tags.tag_id.isin(matched_tags_id), "matched"] = 1

        return events, tags, match_df

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

    def get_proportions2(self, x, by, col="", total=0):
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
            res["lbl_" + col] = "n = {}\n".format(total)
            # res["lbl_" + col] = "n = {}\n({}%)\n\n".format(
            #     x.shape[0], round(x.shape[0] / total * 100, 2)
            # )
        return res

    @staticmethod
    def get_matched_label(x, n_total, n_matched):
        x = int(x)
        label = x == 1 and "matched" or "unmatched"
        label += " (n={}, {}%)".format(
            n_matched[x], round(n_matched[x] / n_total * 100, 1)
        )
        return label

    def plot_tag_repartition(self, data, options):
        tag_df = data["tags"]
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
                mapping=aes(label="lbl_background"),
                position=position_dodge(width=0.9),
            )
            + geom_text(
                mapping=aes(
                    y=max(tags_summary.n_tags) + 30,
                    label="lbl_matched",
                )
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

    def plot_detected_tags(self, data, options):
        tmp = data["tags"][["tag", "matched", "tag_id"]].copy()
        tmp.loc[:, "prop_matched"] = -1
        tmp.loc[:, "n_tags"] = -1

        # Get all proportions bu tags
        tags_summary = (
            tmp.groupby("tag")
            .apply(self.get_proportions, by=["matched"])
            .reset_index(drop=True)
        )
        # convert tags to category
        tags_summary.tag = tags_summary.tag.astype("category")
        # Get list of all tags
        all_tags = tags_summary.tag.unique()
        matched = tags_summary.loc[tags_summary.matched == 1]
        # Get list of all matched tags
        matched_tags = matched.tag.unique()
        unmatched = []
        # Add a row for all unmatched tags
        for tag in all_tags:
            if tag not in matched_tags:
                row = tags_summary.loc[
                    (tags_summary.matched == 0) & (tags_summary.tag == tag)
                ].copy()
                row.prop_matched = 0
                unmatched.append(row)
        unmatched.append(matched)
        # Create final object with unmatched and matched tags
        m_df = pd.concat(unmatched)

        # Sort values
        m_df = m_df.sort_values(["prop_matched", "tag"])
        m_df.tag = m_df.tag.cat.reorder_categories(m_df.tag.to_list())

        plt = ggplot(
            data=m_df,
            mapping=aes(
                x="tag",
                y="prop_matched",
                fill="tag",  # "factor(species, ordered=False)",
            ),
        )
        plot_width = 10 + len(m_df.tag.unique()) * 0.75
        plt = (
            plt
            + geom_bar(stat="identity", show_legend=True, position=position_dodge())
            + xlab("Species")
            + ylab("Proportion of annotation matched")
            + geom_text(
                mapping=aes(label="lbl_matched"),
                position=position_dodge(width=0.9),
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
                    "Proportion of tags detected for model {}, database {}, class {}\n"
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

    def plot_overlap_duration(self, data, options):
        matches = data["matches"]
        matches = matches.loc[matches.tag_overlap > 0]
        # matches.loc[:, "log_dur"] = log()

        plt = ggplot(
            data=matches,
            mapping=aes(
                x="tag_duration",
                y="tag_overlap",
            ),
        )
        plt = (
            plt
            + geom_point()
            + xlab("Tag duration")
            + ylab("Proportion tag overlapping with matching event")
            + theme_classic()
            + theme(
                axis_text_x=element_text(angle=90, vjust=1, hjust=1, margin={"r": -30}),
                plot_title=element_text(
                    weight="bold", size=14, margin={"t": 10, "b": 10}
                ),
                figure_size=(10, 10),
                text=element_text(size=12, weight="bold"),
            )
            + ggtitle(
                (
                    "Proportion of tag overlapping with matching event depending on duration "
                    + "size for model {}, database {}, class {}\n"
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

    def plot_overlap_duration_bar(self, data, options):
        matches = data["matches"]
        matches = matches.loc[matches.tag_overlap > 0]
        matches.loc[:, "tag_overlap_bin"] = pd.cut(
            matches.tag_overlap, [0, 0.25, 0.5, 0.75, 1]
        )
        matches.loc[:, "tag_duration_bin"] = pd.cut(
            matches.tag_duration, [0, 0.25, 0.5, 0.75, 1, 1.5, 2, float("inf")]
        )

        matches.loc[matches.tag_overlap < 0.3].to_csv("small_overlap.csv")

        # matches.loc[:, "log_dur"] = log()

        plt = ggplot(
            data=matches,
            mapping=aes(
                x="tag_duration_bin",
                fill="tag_overlap_bin",
            ),
        )
        plt = (
            plt
            + geom_bar()
            + xlab("Tag duration")
            + ylab("Proportion tag overlapping with matching event")
            + theme_classic()
            + theme(
                axis_text_x=element_text(angle=90, vjust=1, hjust=1, margin={"r": -30}),
                plot_title=element_text(
                    weight="bold", size=14, margin={"t": 10, "b": 10}
                ),
                figure_size=(10, 10),
                text=element_text(size=12, weight="bold"),
            )
            + ggtitle(
                (
                    "Proportion of tag overlapping with matching event depending on duration "
                    + "size for model {}, database {}, class {}\n"
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

    def get_stats(self, events, tags, matches, options):

        n_events = events.shape[0]
        n_tags = tags.shape[0]

        n_true_positives = events.matched.sum()
        n_false_positives = n_events - n_true_positives

        n_tags_matched = tags.matched.sum()
        n_tags_unmatched = n_tags - n_tags_matched

        precision = round(n_true_positives / n_events, 3) if n_events else 0
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

        stats = {
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
            "prop_tag_overlap_75": (
                sum(matches.tag_overlap > 0.75) / matches.shape[0] * 100
            ),
            "mean_event_overlap": round(matches.event_overlap.mean(), 2),
            "mean_tag_overlap": round(matches.tag_overlap.mean(), 2),
            "IoU": round(matches.iou.mean(), 2),
        }

        print("Stats for options {0}:".format(options))
        common_utils.print_warning(
            "Precision:{}; Recall:{}; F1_score:{}; IoU:{}; Mean event overlap:{}, Mean tag overlap:{}".format(
                stats["precision"],
                stats["recall"],
                stats["f1_score"],
                stats["IoU"],
                stats["mean_event_overlap"],
                stats["mean_tag_overlap"],
            )
        )

        return pd.DataFrame([stats])

    def evaluate(self, data, options, infos):
        predictions, tags = data
        tags = tags["tags_df"]
        events = self.filter_predictions(predictions, options)
        events, tags, matches = self.get_matches(events, tags, options)
        stats = self.get_stats(events, tags, matches, options)

        res = {
            "stats": stats,
            "matches": matches,
        }
        if options.get("draw_plots", False):
            res["plots"] = self.draw_plots(
                data={"events": events, "tags": tags, "matches": matches},
                options=options,
            )
        return res

    # def plot_PR_curve(self, stats, options):
    #     return stats
