from cmath import nan
import math

import matplotlib.pyplot as plt
import pandas as pd
from mouffet import common_utils
from mouffet.evaluation import Evaluator
from sklearn import metrics

from plotnine import (
    aes,
    element_text,
    xlim,
    ylim,
    annotate,
    geom_line,
    ggplot,
    ggtitle,
    theme,
    theme_classic,
    xlab,
    ylab,
)


class DirectEvaluator(Evaluator):

    NAME = "direct"

    REQUIRES = ["tags_df"]

    DEFAULT_ACTIVITY_THRESHOLD = 0.7

    DEFAULT_TIME_BUFFER = 0.5

    DEFAULT_REFINE_OPTIONS = {"min_duration": 300, "activity_threshold_min": 0.1}

    def file_event_duration(self, df):
        if df.shape[0] > 0:
            step = df.time.iloc[1] - df.time.iloc[0]
            return df.events.sum() / 2 * step
        return 0

    def file_tag_duration(self, df):
        if df.shape[0] > 1:
            step = df.time.iloc[1] - df.time.iloc[0]
            return df.tags.sum() * step
        return 0

    def refine_predictions(self, predictions, options):
        step = predictions.time.values[1] - predictions.time.values[0]
        refine_options = self.DEFAULT_REFINE_OPTIONS
        refine_options.update(options.get("refine_options", {}))
        event_start = 0
        for i in range(0, predictions.shape[0]):
            # * no event started, start one
            if predictions.events.iloc[i] > 0 and event_start == 0:
                event_start = i
            # * event is ending
            elif predictions.events.iloc[i] == 0 and event_start > 0:
                event_end = i - 1
                event_duration = (event_end - event_start) * step * 1000
                if event_duration <= refine_options["min_duration"] or (
                    max(
                        predictions.activity.iloc[event_start:event_end]
                        < refine_options["activity_threshold_min"]
                    )
                ):
                    # if predictions.tags.iloc[event_start:event_end].sum() > 0:
                    #     print(predictions.iloc[event_start:event_end])
                    if event_start == event_end:
                        predictions.events.iloc[event_start] = 0
                        # predictions.activity.iloc[event_start] = (
                        #     predictions.activity.iloc[event_start] - 0.3
                        # )
                    else:
                        predictions.events.iloc[event_start:event_end] = 0
                        # predictions.activity.iloc[event_start:event_end] = (
                        #     predictions.activity.iloc[event_start] - 0.3
                        # )
                # TODO: remove buffer?
                event_start = 0

        return predictions

    def filter_predictions(self, predictions, options, tags=None):
        predictions = predictions[["activity", "recording_id", "time"]].copy()
        predictions["events"] = 0
        predictions["tags"] = 0
        threshold = options.get("activity_threshold", self.DEFAULT_ACTIVITY_THRESHOLD)
        predictions.loc[predictions.activity > threshold, "events"] = 2
        if tags is not None and not tags.empty:
            events = predictions.groupby("recording_id", as_index=True, observed=True)
            predictions = events.apply(self.get_tags_presence, tags, options)
        if options.get("refine_predictions", False):
            events = predictions.groupby("recording_id", as_index=True, observed=True)
            predictions = events.apply(self.refine_predictions, options)
        return predictions

    def get_tags_presence(self, predictions, tags, options=None):
        if not predictions.empty:
            time_buffer = options.get("time_buffer", self.DEFAULT_TIME_BUFFER)
            options = options or {}
            tags = tags[tags.recording_id == predictions.name]
            # threshold = options.get("activity_threshold", self.DEFAULT_ACTIVITY_THRESHOLD)
            # predictions.loc[predictions.activity > threshold, "events"] = 2
            dur = max(predictions.time)
            step = predictions.time.values[1] - predictions.time.values[0]
            for tag in tags.itertuples():
                start = max(0, math.ceil((tag.tag_start - time_buffer) / step))
                end = min(
                    math.ceil(dur / step),
                    math.ceil((tag.tag_end + time_buffer) / step),
                )
                predictions.tags.iloc[start:end] = 1
        return predictions

    def calculate_stats(self, predictions, options):
        # predictions.loc[predictions.activity > threshold, "events"] = 2

        res = predictions.tags + predictions.events

        n_true_positives = len(res[res == 3])
        n_true_negatives = len(res[res == 0])
        n_false_positives = len(res[res == 2])
        n_false_negatives = len(res[res == 1])

        intersection = predictions.tags * predictions.events  # Logical AND
        union = predictions.tags + predictions.events

        iou = round(intersection.sum() / float(union.sum()), 3)

        unbalanced_accuracy = round(
            (float(n_true_positives + n_true_negatives) / predictions.shape[0]), 3
        )

        A = float(n_true_positives) / (n_true_positives + n_false_negatives)
        B = float(n_true_negatives) / (n_false_positives + n_true_negatives)
        balanced_accuracy = round((A + B) / 2.0, 3)

        precision = round(n_true_positives / (n_true_positives + n_false_positives), 3)
        recall = round(n_true_positives / (n_true_positives + n_false_negatives), 3)
        f1_score = round(2 * precision * recall / (precision + recall), 3)

        average_precision = round(
            metrics.average_precision_score(predictions.tags, predictions.activity), 3
        )
        auc = round(metrics.roc_auc_score(predictions.tags, predictions.activity), 3)

        stats = {
            "activity_threshold": options.get(
                "activity_threshold", self.DEFAULT_ACTIVITY_THRESHOLD
            ),
            "n_true_positives": n_true_positives,
            "n_false_positives": n_false_positives,
            "n_true_negatives": n_true_negatives,
            "n_false_negatives": n_false_negatives,
            "unbalanced_accuracy": unbalanced_accuracy,
            "balanced_accuracy": balanced_accuracy,
            "average_precision": average_precision,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "auc": auc,
            "ap": average_precision,
            "IoU": iou,
        }

        common_utils.print_warning(
            # f"Threshold: {threshold},"
            f'Precision: { stats["precision"]};'
            + f' Recall: {stats["recall"]};'
            + f' F1_score: {stats["f1_score"]};'
            + f' AUC: {stats["auc"]};'
            + f' Average precision: {stats["ap"]};'
            + f' IoU: {stats["IoU"]}'
        )
        return pd.DataFrame([stats])  # , predictions

    def get_stats(self, predictions, options):

        predictions.activity = predictions.activity.fillna(0)

        # stats = []
        # matches = []
        # threshold = options.get("activity_threshold", self.DEFAULT_ACTIVITY_THRESHOLD)
        # if threshold == "multiple":
        #     threshold_by = options.get("threshold_by", 0.1)
        #     threshold_start = options.get("threshold_start", 0)
        #     threshold_end = options.get("threshold_end", 0.9)
        #     thresholds = common_utils.range_list(
        #         threshold_start, threshold_end, step=threshold_by
        #     )
        # else:
        #     thresholds = [threshold]

        print("Stats for options {0}:".format(options))

        return self.calculate_stats(predictions.copy(), options)
        # for thresh in thresholds:
        #     tmp_stats, tmp_matches = self.calculate_stats(
        #         predictions.copy(), thresh, options
        #     )
        #     stats.append(tmp_stats)
        #     matches.append(tmp_matches)

        # stats = pd.DataFrame(stats)

        # return stats, matches

    def plot_pr_curve(self, data, options, infos):
        events = data["events"]
        precision, recall, _ = metrics.precision_recall_curve(
            events.tags, events.activity
        )
        plt_data = pd.DataFrame({"precision": precision, "recall": recall})
        plt = (
            ggplot(
                data=plt_data,
                mapping=aes(
                    x="precision",  # "factor(species, ordered=False)",
                    y="recall",
                ),
            )
            + geom_line()
            + xlim([0, 1])
            + ylim([0, 1])
            + xlab("Precision")
            + ylab("Recall")
            + annotate(
                "text",
                x=0.75,
                y=0.75,
                label="Average Precision: {}".format(data["stats"]["ap"].iloc[0]),
            )
            + theme_classic()
            + theme(
                plot_title=element_text(
                    weight="bold", size=14, margin={"t": 10, "b": 10}
                ),
                text=element_text(size=12, weight="bold"),
            )
            + ggtitle(
                ("Precision-Recall curve for model {} on database {}").format(
                    options["scenario_info"]["model"],
                    options["scenario_info"]["database"],
                )
            )
        )

        return plt

    def plot_roc(self, data, options, infos):
        events = data["events"]
        fpr, tpr, _ = metrics.roc_curve(events.tags, events.activity)
        plt_data = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        plt = (
            ggplot(
                data=plt_data,
                mapping=aes(
                    x="fpr",  # "factor(species, ordered=False)",
                    y="tpr",
                ),
            )
            + geom_line()
            + xlim([0, 1])
            + ylim([0, 1])
            + xlab("False positive rate")
            + ylab("True positive rate")
            + annotate(
                "text",
                x=0.75,
                y=0.75,
                label="AUC: {}".format(data["stats"]["auc"].iloc[0]),
            )
            + theme_classic()
            + theme(
                plot_title=element_text(
                    weight="bold", size=14, margin={"t": 10, "b": 10}
                ),
                text=element_text(size=12, weight="bold"),
            )
            + ggtitle(
                (
                    "Receiving operator characteristic curve for model {} on database {}"
                ).format(
                    options["scenario_info"]["model"],
                    options["scenario_info"]["database"],
                )
            )
        )

        return plt

    def evaluate(self, data, options, infos):
        predictions, tags = data
        tags_df = tags["tags_df"]
        events = self.filter_predictions(predictions, options, tags_df)
        stats = self.get_stats(events, options)
        res = {"stats": stats, "matches": events}
        if options.get("draw_plots", False):
            res["plots"] = self.draw_plots(
                data={"events": events, "tags": tags, "stats": stats},
                options=options,
                infos=infos,
            )
        return res
