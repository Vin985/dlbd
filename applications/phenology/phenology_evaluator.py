from pathlib import Path

import pandas as pd
from dlbd.data.tag_utils import flatten_tags
from dlbd.evaluation import EVALUATORS
from mouffet import common_utils
from mouffet.evaluation import Evaluator
from plotnine import (
    aes,
    facet_grid,
    geom_line,
    geom_point,
    geom_smooth,
    ggplot,
    ggtitle,
    save_as_pdf_pages,
)
from plotnine.labels import ggtitle
from scipy.spatial.distance import euclidean
from sklearn import metrics
from statsmodels.tsa.seasonal import seasonal_decompose

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

    def file_event_duration(self, df):
        # * Returns total duration of events in a file
        if df.empty:
            return 0
        return df.drop_duplicates("event_id")["event_duration"].sum()

    def file_tag_duration(self, df):
        flattened = flatten_tags(df.drop_duplicates("tag_id"))
        return flattened.tag_duration.sum()

    def daily_mean_activity(self, df):
        # * Returns mean activity duration in a day
        return df["event_duration"].mean()

    def get_daily_activity(self, df, options, df_type="event"):
        res = {}
        # * Get total duration per file
        file_song_duration = (
            df.groupby("file_name")
            .apply(getattr(self, "file_" + df_type + "_duration"))
            .reset_index()
            .rename(columns={0: "total_duration"})
        )
        file_song_duration[
            ["site", "plot", "date", "time", "to_drop"]
        ] = file_song_duration.file_name.str.split("_", expand=True)
        file_song_duration = file_song_duration.assign(
            full_date=[
                str(x) + "_" + y
                for x, y in zip(file_song_duration["date"], file_song_duration["time"])
            ]
        )
        file_song_duration["full_date"] = pd.to_datetime(
            file_song_duration["full_date"], format="%Y-%m-%d_%H%M%S"
        )
        file_song_duration["date"] = pd.to_datetime(
            file_song_duration["date"], format="%Y-%m-%d"
        )
        file_song_duration = file_song_duration.drop(columns=["to_drop", "file_name"])
        res["file_duration"] = file_song_duration

        # * Get mean duration per day
        daily_duration = (
            file_song_duration[["date", "total_duration"]]
            .groupby("date")
            .agg("mean")
            .dropna()
        )

        trend = seasonal_decompose(
            daily_duration, model="additive", extrapolate_trend="freq"
        ).trend.reset_index(name="trend")

        daily_duration = daily_duration.reset_index().merge(trend)

        daily_duration["trend_norm"] = (
            daily_duration.trend - daily_duration.trend.mean()
        ) / daily_duration.trend.std()

        if df_type == "tag":
            daily_duration["type"] = "ground_truth"
        else:
            daily_duration["type"] = options["scenario_info"]["model"]

        res["daily_duration"] = daily_duration

        return res

    def plot_distances(self, data, options):
        plt_df = data["df"]
        tmp_plt = (
            ggplot(
                data=plt_df,
                mapping=aes("date", "trend_norm", color="type"),
            )
            + geom_line()
            + ggtitle("Distance: {}".format(data["distance"]))
        )

        return tmp_plt
        # norm_plt = (
        #     ggplot(
        #         data=trend.dropna(),
        #         mapping=aes(
        #             "date",
        #             "norm",
        #         ),
        #     )
        #     + geom_line(colour="#ff0000")
        #     + geom_point(data=peaks_df, mapping=aes("date", "norm"), colour="#00ff00")
        #     + geom_line(
        #         data=ref,
        #         mapping=aes("date", "norm"),
        #         colour="#0000ff",
        #     )
        #     + ggtitle(model)
        # )
        # plots.append(tmp_plt)
        # norm_plots.append(norm_plt)

    def evaluate(self, predictions, tags, options):
        method = options["method"]
        stats = EVALUATORS[method].evaluate(predictions, tags, options)

        matches = stats["matches"]

        daily_tags_duration = self.get_daily_activity(matches, options, "tag")
        daily_events_duration = self.get_daily_activity(matches, options, "event")

        eucl_distance = round(
            euclidean(
                daily_tags_duration["daily_duration"].trend_norm,
                daily_events_duration["daily_duration"].trend_norm,
            ),
            3,
        )

        common_utils.print_warning(
            "Distance for model {}: eucl: {}".format(
                options["scenario_info"]["model"], eucl_distance
            )
        )

        stats["stats"].loc[:, "eucl_distance"] = eucl_distance
        stats["tags_duration"] = daily_tags_duration
        stats["events_duration"] = daily_events_duration

        if options.get("draw_plots", False):
            stats["plots"] = self.draw_plots(
                data={
                    "df": pd.concat(
                        [
                            daily_tags_duration["daily_duration"],
                            daily_events_duration["daily_duration"],
                        ]
                    ),
                    "distance": eucl_distance,
                },
                options=options,
            )

        return stats
