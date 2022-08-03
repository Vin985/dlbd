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


class CityNetEvaluator(Evaluator):

    NAME = "citynet"

    REQUIRES = ["tags_df"]

    DEFAULT_ACTIVITY_THRESHOLD = 0.5

    DEFAULT_TIME_BUFFER = 0.5

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

    def filter_predictions(self, predictions, options, tags=None):
        predictions = predictions[["activity", "recording_id", "time"]].copy()
        predictions["events"] = 0
        predictions["tags"] = 0
        events = predictions.groupby("recording_id", as_index=True, observed=True)
        events = events.apply(self.get_recording_events, tags, options)
        return events

    def get_recording_events(self, predictions, tags, options=None):
        options = options or {}
        if tags is not None and not tags.empty:
            tags = tags[tags.recording_id == predictions.name]
        threshold = options.get("activity_threshold", self.DEFAULT_ACTIVITY_THRESHOLD)
        predictions.loc[predictions.activity > threshold, "events"] = 2
        dur = max(predictions.time)
        if not predictions.empty:
            step = predictions.time.values[1] - predictions.time.values[0]
            if tags is not None and not tags.empty:
                for tag in tags.itertuples():
                    start = max(
                        0, math.ceil((tag.tag_start - self.DEFAULT_TIME_BUFFER) / step)
                    )
                    end = min(
                        math.ceil(dur / step),
                        math.ceil((tag.tag_end + self.DEFAULT_TIME_BUFFER) / step),
                    )
                    predictions.tags.iloc[start:end] = 1
        return predictions

    def get_stats(self, predictions, options):

        res = predictions.tags + predictions.events

        n_true_positives = len(res[res == 3])
        n_true_negatives = len(res[res == 0])
        n_false_positives = len(res[res == 2])
        n_false_negatives = len(res[res == 1])

        unbalanced_accuracy = round(
            (float(n_true_positives + n_true_negatives) / predictions.shape[0]), 3
        )

        A = float(n_true_positives) / (n_true_positives + n_false_negatives)
        B = float(n_true_negatives) / (n_false_positives + n_true_negatives)
        balanced_accuracy = round((A + B) / 2.0, 3)

        precision = round(n_true_positives / (n_true_positives + n_false_positives), 3)
        recall = round(n_true_positives / (n_true_positives + n_false_negatives), 3)
        average_precision = round(
            metrics.average_precision_score(predictions.tags, predictions.activity), 3
        )
        auc = round(metrics.roc_auc_score(predictions.tags, predictions.activity), 3)

        f1_score = round(2 * precision * recall / (precision + recall), 3)

        stats = {
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
        }

        print("Stats for options {0}:".format(options))
        common_utils.print_warning(
            "Precision: {}; Recall: {}; F1_score: {}; AUC: {}; Average precision: {}".format(
                stats["precision"],
                stats["recall"],
                stats["f1_score"],
                stats["auc"],
                stats["ap"],
            )
        )
        return pd.DataFrame([stats])

    def get_precision_recall_curve(self, predictions, prec, rec):

        precision, recall, _ = metrics.precision_recall_curve(
            predictions.tags, predictions.activity
        )
        plt.plot(recall, precision)
        plt.plot(rec, prec, "ob", ms=6)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.draw()
        plt.savefig("pr_curve.pdf")

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
                ("Precision recall curve for database {}").format(
                    options["scenario_info"]["database"],
                )
            )
        )

        return plt

    def evaluate(self, data, options, infos):
        predictions, tags = data
        tags = tags["tags_df"]
        events = self.filter_predictions(predictions, options, tags)
        stats = self.get_stats(events, options)
        res = {"stats": stats, "matches": events}
        if options.get("draw_plots", False):
            res["plots"] = self.draw_plots(
                data={"events": events, "tags": tags, "stats": stats},
                options=options,
                infos=infos,
            )
        return res
