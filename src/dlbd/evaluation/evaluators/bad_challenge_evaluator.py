import math

import matplotlib
import pandas as pd
from mouffet.evaluation.evaluator import Evaluator

import matplotlib.pyplot as plt
from mouffet.utils import common as common_utils
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn import metrics

# matplotlib.use("agg")


class BADChallengeEvaluator(Evaluator):

    REQUIRES = ["tags_df"]

    DEFAULT_EVENT_THRESHOLD = 0.5

    DEFAULT_TIME_BUFFER = 0.5

    def __init__(self):
        super().__init__()

    def get_events(self, predictions, tags, options):
        predictions = predictions[["activity", "recording_id", "time"]].copy()
        predictions["events"] = 0
        predictions["tags"] = 0
        events = predictions.groupby("recording_id", as_index=True, observed=True)
        events = events.apply(self.get_recording_events, tags, options)
        return events

    def get_recording_events(self, predictions, tags, options=None):
        options = options or {}
        tags = tags[tags.recording_id == predictions.name]
        threshold = options.get("event_threshold", self.DEFAULT_EVENT_THRESHOLD)
        predictions.loc[predictions.activity > threshold, "events"] = 2
        dur = max(predictions.time)
        if not predictions.empty:
            step = predictions.time.values[1] - predictions.time.values[0]
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
            average_precision_score(predictions.tags, predictions.activity), 3
        )
        auc = round(metrics.roc_auc_score(predictions.tags, predictions.activity), 3)

        # self.get_precision_recall_curve(predictions, precision, recall)

        f1_score = round(2 * precision * recall / (precision + recall), 3)

        stats = {
            # "n_events": events.shape[0],
            # "n_tags": tags.shape[0],
            "n_true_positives": n_true_positives,
            "n_false_positives": n_false_positives,
            "n_true_negatives": n_true_negatives,
            "n_false_negatives": n_false_negatives,
            "unbalanced_accuracy": unbalanced_accuracy,
            "balanced_accuracy": balanced_accuracy,
            "average_precision": average_precision,
            # "n_tags_matched": n_tags_matched,
            # "n_tags_unmatched": n_tags_unmatched,
            # "true_positives_ratio": true_positives_ratio,
            # "false_positive_rate": false_positive_rate,
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

        precision, recall, _ = precision_recall_curve(
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

    def evaluate(self, predictions, tags, options):
        tags = tags["tags_df"]
        events = self.get_events(predictions, tags, options)
        stats = self.get_stats(events, options)
        res = {"stats": stats, "matches": events}
        if options.get("draw_plots", False):
            res["plots"] = self.draw_plots(
                data={"events": events, "tags": tags},
                options=options,
            )
        return res
