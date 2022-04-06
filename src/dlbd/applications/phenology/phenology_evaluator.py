import pandas as pd
from ...utils.plot_utils import format_date_short
from mouffet import common_utils
from mouffet.evaluation import Evaluator
from pandas_path import path  # pylint: disable=unused-import
from plotnine import *
from scipy.spatial.distance import euclidean
from statsmodels.tsa.seasonal import seasonal_decompose

from ...evaluation import EVALUATORS


class PhenologyEvaluator(Evaluator):

    REQUIRES = ["tags_df"]

    def file_event_duration(self, df, method):
        return EVALUATORS[method].file_event_duration(df)

    def file_tag_duration(self, df, method):
        return EVALUATORS[method].file_tag_duration(df)

    def daily_mean_activity(self, df):
        # * Returns mean activity duration in a day
        return df["event_duration"].mean()

    def extract_recording_info_dlbd(self, df, options):
        df[
            ["site", "plot", "date", "time", "to_drop"]
        ] = df.recording_id.path.stem.str.split("_", expand=True)
        df = df.assign(
            full_date=[str(x) + "_" + y for x, y in zip(df["date"], df["time"])]
        )
        df["full_date"] = pd.to_datetime(df["full_date"], format="%Y-%m-%d_%H%M%S")
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df = df.drop(columns=["to_drop", "recording_id"])
        return df

    def extract_recording_info_audiomoth2019(self, df, options):
        df[["date", "time"]] = df.recording_id.path.stem.str.split("_", expand=True)
        df = df.assign(
            full_date=[str(x) + "_" + y for x, y in zip(df["date"], df["time"])]
        )
        df["full_date"] = pd.to_datetime(df["full_date"], format="%Y%m%d_%H%M%S")
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        return df

    def get_daily_activity(self, df, options, df_type="event"):
        res = {}
        # * Get total duration per file
        file_song_duration = (
            df.groupby("recording_id")
            .apply(
                getattr(self, "file_" + df_type + "_duration"), method=options["method"]
            )
            .reset_index()
            .rename(columns={0: "total_duration"})
        )
        file_song_duration = getattr(
            self, "extract_recording_info_" + options.get("recording_info_type", "dlbd")
        )(file_song_duration, options=options)
        res["file_duration"] = file_song_duration

        # * Get mean duration per day
        daily_duration = (
            file_song_duration[["date", "total_duration"]]
            .groupby("date")
            .agg("mean")
            .dropna()
        )

        trend = (
            seasonal_decompose(
                daily_duration,
                model="additive",
            )
            .trend.reset_index(name="trend")
            .dropna()
        )

        daily_duration = daily_duration.reset_index().merge(trend)

        daily_duration["trend_norm"] = (
            daily_duration.trend - daily_duration.trend.mean()
        ) / daily_duration.trend.std()

        if df_type == "tag":
            daily_duration["type"] = "ground_truth"
        elif options.get("scenario_info"):
            daily_duration["type"] = options["scenario_info"]["model"]
        else:
            daily_duration["type"] = options.get("plot", "unknown_plot")

        res["daily_duration"] = daily_duration

        return res

    def plot_distances(self, data, options):
        plt_df = data["df"]
        res = []
        if options.get("plot_real_distance", True):
            tmp_plt = (
                ggplot(
                    data=plt_df,
                    mapping=aes("date", "trend", color="type"),
                )
                + geom_line()
                + ggtitle(
                    "Daily mean activity per recording.\n"
                    + " Euclidean distance to reference: {}".format(data["distance"])
                )
                + xlab("Date")
                + ylab("Daily mean activity per recording (s)")
                + scale_color_discrete(labels=["Model", "Reference"])
                + scale_x_datetime(labels=format_date_short)
                + theme_classic()
                + theme(axis_text_x=element_text(angle=45))
            )
            res.append(tmp_plt)
        if options.get("plot_norm_distance", True):
            tmp_plt_norm = (
                ggplot(
                    data=plt_df,
                    mapping=aes("date", "trend_norm", color="type"),
                )
                + geom_line()
                + ggtitle(
                    "Normalized daily mean activity per recording.\n"
                    + " Euclidean distance to reference: {}".format(
                        data["distance_norm"]
                    )
                )
                + xlab("Date")
                + ylab("Normalized daily mean activity per recording")
                + scale_color_discrete(labels=["Model", "Reference"])
                + scale_x_datetime(labels=format_date_short)
                + theme_classic()
                + theme(axis_text_x=element_text(angle=45))
            )
            res.append(tmp_plt_norm)

        return res

    def evaluate(self, predictions, tags, options):
        if options["scenario_info"]["database"] not in options.get(
            "phenology_databases", []
        ):
            common_utils.print_info(
                (
                    "Database {} is not part of the accepted databases for the 'Phenology' "
                    + "evaluator described in the 'phenology_databases' option. Skipping."
                ).format(options["scenario_info"]["database"])
            )
            return {}
        method = options["method"]
        stats = EVALUATORS[method].evaluate(predictions, tags, options)

        matches = stats["matches"]

        daily_tags_duration = self.get_daily_activity(matches, options, "tag")
        daily_events_duration = self.get_daily_activity(matches, options, "event")

        eucl_distance = round(
            euclidean(
                daily_tags_duration["daily_duration"].trend,
                daily_events_duration["daily_duration"].trend,
            ),
            3,
        )
        eucl_distance_norm = round(
            euclidean(
                daily_tags_duration["daily_duration"].trend_norm,
                daily_events_duration["daily_duration"].trend_norm,
            ),
            3,
        )

        common_utils.print_warning(
            "Distance for model {}: eucl: {}; eucl_norm: {}".format(
                options["scenario_info"]["model"], eucl_distance, eucl_distance_norm
            )
        )

        stats["stats"].loc[:, "eucl_distance"] = eucl_distance
        stats["stats"].loc[:, "eucl_distance_norm"] = eucl_distance_norm
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
                    "distance_norm": eucl_distance_norm,
                    "method": method,
                },
                options=options,
            )

        return stats
