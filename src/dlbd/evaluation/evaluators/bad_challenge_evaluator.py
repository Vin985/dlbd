import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mouffet.evaluation.evaluator import Evaluator
from mouffet.utils import common as common_utils
from sklearn import metrics
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

from .citynet_evaluator import CityNetEvaluator
from .standard_evaluator import StandardEvaluator
from .subsampling_evaluator import SubsamplingEvaluator

# matplotlib.use("agg")


class BADChallengeEvaluator(Evaluator):

    EVALUATORS = {
        "standard": StandardEvaluator(),
        "subsampling": SubsamplingEvaluator(),
        "citynet": CityNetEvaluator(),
    }

    REQUIRES = ["tags_df"]

    DEFAULT_EVENT_THRESHOLD = 0.5

    DEFAULT_TIME_BUFFER = 0.5

    def __init__(self):
        super().__init__()

    def has_bird_standard(self, events):
        if not events.empty:
            return 2
        return 0

    def get_events(self, predictions, options):
        method = options["method"]
        events = self.EVALUATORS[method].get_events(predictions, options)
        agg_func = getattr(self, "has_bird_" + method)
        res = []
        for recording in predictions.recording_id.unique():
            tmp = {}
            rec_events = events.loc[events.recording_id == recording]
            tmp["itemid"] = Path(recording).stem
            tmp["events"] = agg_func(rec_events)
            res.append(tmp)
        res_df = pd.DataFrame(res)
        return res_df

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
        # average_precision = round(
        #     average_precision_score(predictions.tags, predictions.activity), 3
        # )
        # auc = round(metrics.roc_auc_score(predictions.tags, predictions.activity), 3)

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
            # "average_precision": average_precision,
            # "n_tags_matched": n_tags_matched,
            # "n_tags_unmatched": n_tags_unmatched,
            # "true_positives_ratio": true_positives_ratio,
            # "false_positive_rate": false_positive_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            # "auc": auc,
            # "ap": average_precision,
        }

        print("Stats for options {0}:".format(options))
        common_utils.print_warning(
            "Precision: {}; Recall: {}; F1_score: {}".format(
                stats["precision"],
                stats["recall"],
                stats["f1_score"],
                # stats["auc"],
                # stats["ap"],
            )
        )
        return pd.DataFrame([stats])

    def evaluate(self, predictions, tags, options):
        events = self.get_events(predictions, options)
        tags = tags["tags_df"]
        tags = tags.rename(columns={"hasbird": "tags"})
        events = events.merge(tags)
        stats = self.get_stats(events, options)
        res = {"stats": stats, "matches": events}
        if options.get("draw_plots", False):
            res["plots"] = self.draw_plots(
                data={"events": events, "tags": tags},
                options=options,
            )
        return res
