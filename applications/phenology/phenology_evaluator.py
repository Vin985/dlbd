from pathlib import Path

import pandas as pd
from dlbd.evaluation import EVALUATORS
from mouffet import common_utils
from mouffet.evaluation import Evaluator
from sklearn import metrics

# matplotlib.use("agg")


class PhenologyEvaluator(Evaluator):

    REQUIRES = ["tags_df"]

    DEFAULT_EVENT_THRESHOLD = 0.5

    DEFAULT_TIME_BUFFER = 0.5

    def has_bird_standard(self, events):
        if not events.empty:
            return 2
        return 0

    def has_bird_simple(self, predictions, options):
        tmp = predictions.groupby("recording_id").agg({"activity": "max"}).reset_index()
        tmp["itemid"] = tmp.recording_id.path.stem
        tmp = tmp.drop(columns="recording_id")
        tmp.loc[tmp.activity >= options["activity_threshold"], "events"] = 2
        return tmp

    def get_events(self, predictions, options):
        method = options["method"]
        if method == "simple":
            res_df = self.has_bird_simple(predictions, options)
        else:
            events = EVALUATORS[method].get_events(predictions, options)
            agg_func = getattr(self, "has_bird_" + method)
            res = []
            for recording in predictions.recording_id.unique():
                # * We iterate instead of groupby in case no events are detected
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

        average_precision = None
        auc = None

        unbalanced_accuracy = round(
            (float(n_true_positives + n_true_negatives) / predictions.shape[0]), 3
        )

        A = float(n_true_positives) / (n_true_positives + n_false_negatives)
        B = float(n_true_negatives) / (n_false_positives + n_true_negatives)
        balanced_accuracy = round((A + B) / 2.0, 3)

        precision = round(n_true_positives / (n_true_positives + n_false_positives), 3)
        recall = round(n_true_positives / (n_true_positives + n_false_negatives), 3)
        if options["method"] == "simple":
            average_precision = round(
                metrics.average_precision_score(predictions.tags, predictions.activity),
                3,
            )
            auc = round(
                metrics.roc_auc_score(predictions.tags, predictions.activity), 3
            )

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
            "auc": auc,
            "ap": average_precision,
        }

        print("Stats for options {0}:".format(options))
        common_utils.print_warning(
            "Precision: {}; Recall: {}; F1_score: {}; AUC: {}; mAP: {}".format(
                stats["precision"],
                stats["recall"],
                stats["f1_score"],
                stats["auc"],
                stats["ap"],
            )
        )
        return pd.DataFrame([stats])

    def file_events_duration(self, df):
        # * Returns total duration of events in a file
        if df.empty:
            return 0
        return df.drop_duplicates("event_id")["event_duration"].sum()

    def daily_mean_activity(self, df):
        # * Returns mean activity duration in a day
        return df["event_duration"].mean()

    def evaluate(self, predictions, tags, options):
        res = {}
        method = options["method"]
        stats = EVALUATORS[method].evaluate(predictions, tags, options)

        matches = stats["matches"]

        # * Get total duration per file
        matches = (
            matches[["file_name", "event_id", "event_duration"]]
            .groupby("file_name")
            .apply(self.file_events_duration)
            .reset_index()
            .rename(columns={0: "total_duration"})
        )
        matches[
            ["site", "plot", "date", "time", "to_drop"]
        ] = matches.file_name.str.split("_", expand=True)
        matches = matches.assign(
            full_date=[
                str(x) + "_" + y for x, y in zip(matches["date"], matches["time"])
            ]
        )
        matches["full_date"] = pd.to_datetime(
            matches["full_date"], format="%Y-%m-%d_%H%M%S"
        )
        matches["date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d")
        matches = matches.drop(columns=["to_drop", "file_name"])
        res["full_match"] = matches

        # * Get mean duration per day
        matches_per_day = (
            matches[["date", "total_duration"]]
            .groupby("date")
            .agg("mean")
            .reset_index()
        )
        matches_per_day["type"] = options["scenario_info"]["model"]
        matches_per_day["mov_avg"] = matches_per_day.total_duration.rolling(
            4, center=True
        ).mean()
        # matches_per_day["diff"] = matches_per_day["mov_avg"] - ref_per_day["mov_avg"]
        res["matches_per_day"] = matches_per_day
        return res
