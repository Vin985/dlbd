import pandas as pd
import numpy as np

from .detector import Detector
import math
from time import time
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


class CityNetDetector(Detector):

    DEFAULT_EVENT_THRESHOLD = 0.5

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
        if not predictions.empty:
            step = predictions.time.values[1] - predictions.time.values[0]
            for tag in tags.itertuples():
                start = math.ceil(tag.tag_start / step)
                end = math.ceil(tag.tag_end / step)
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

        f1 = f1_score(predictions.tags, predictions.events == 2)

        precision = round(n_true_positives / (n_true_positives + n_false_positives), 3)
        recall = round(n_true_positives / (n_true_positives + n_false_negatives), 3)
        average_precision = round(
            average_precision_score(predictions.tags, predictions.events), 3
        )

        self.get_precision_recall_curve(predictions, precision, recall)

        f1_2 = round(2 * precision * recall / (precision + recall), 3)
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
            "f1_1": f1,
            "f1_2": f1_2,
        }
        return {
            "stats": stats,
            "predictions": predictions,
            "options": options,
        }

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
        events = self.get_events(predictions, tags, options)
        stats = self.get_stats(events, options)
        print("Stats for options {0}: {1}".format(options, stats["stats"]))
        stats = {}
        return stats
