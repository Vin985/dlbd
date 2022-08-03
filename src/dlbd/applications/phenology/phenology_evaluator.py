from datetime import datetime

import pandas as pd
from mouffet import common_utils
from mouffet.evaluation import Evaluator
from pandas_path import path  # pylint: disable=unused-import
from plotnine import *
from scipy.spatial.distance import euclidean
from statsmodels.tsa.seasonal import seasonal_decompose

from ...evaluation import EVALUATORS
from ...utils.plot_utils import format_date_short
from ...data.tag_utils import flatten_tags


class PhenologyEvaluator(Evaluator):

    NAME = "phenology"

    REQUIRES = ["tags_df"]

    def file_event_duration(self, df, method):
        return EVALUATORS[method].file_event_duration(df)

    def file_tag_duration(self, df, method=None):
        flattened = flatten_tags(df)
        return flattened.tag_duration.sum()

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
        df["date_hour"] = pd.to_datetime(
            df["full_date"].dt.strftime("%Y%m%d_%H"), format="%Y%m%d_%H"
        )
        df = df.drop(columns=["to_drop", "recording_id"])
        return df

    def extract_recording_info_audiomoth2019(self, df, options):
        df[["date", "time"]] = df.recording_id.path.stem.str.split("_", expand=True)
        df = df.assign(
            full_date=[str(x) + "_" + y for x, y in zip(df["date"], df["time"])]
        )
        df["full_date"] = pd.to_datetime(df["full_date"], format="%Y%m%d_%H%M%S")
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df["date_hour"] = pd.to_datetime(
            df["full_date"].dt.strftime("%Y%m%d_%H"), format="%Y%m%d_%H"
        )
        return df

    def extract_recording_info_audiomoth2018(self, df, options):
        df["full_date"] = df.recording_id.path.stem.apply(
            lambda x: datetime.fromtimestamp(int(x, 16))
        )

        df["date"] = pd.to_datetime(df["full_date"].dt.strftime("%Y%m%d"))
        df["date_hour"] = pd.to_datetime(
            df["full_date"].dt.strftime("%Y%m%d_%H"), format="%Y%m%d_%H"
        )
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

        if "depl_start" in options:
            file_song_duration = file_song_duration.loc[
                file_song_duration.date_hour >= options["depl_start"]
            ]
        if "depl_end" in options:
            file_song_duration = file_song_duration.loc[
                file_song_duration.date_hour <= options["depl_end"]
            ]

        res["file_duration"] = file_song_duration

        by_hour = (
            file_song_duration[["date_hour", "total_duration"]]
            .groupby("date_hour")
            .agg("mean")
        )
        by_hour = by_hour.asfreq("H", fill_value=0)

        daily_duration = by_hour.resample("D").agg("mean")
        daily_duration.index.name = "date"

        # # * Get mean duration per day
        # daily_duration = (
        #     file_song_duration[["date", "total_duration"]]
        #     .groupby("date")
        #     .agg("mean")
        #     .dropna()
        # )
        # idx = pd.date_range(daily_duration.index.min(), daily_duration.index.max())
        # idx.name = "date"
        # daily_duration = daily_duration.reindex(idx, fill_value=0)
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

    def plot_distances(self, data, options, infos):
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
                    "Daily mean activity per recording with method {}.\n".format(
                        options["method"]
                    )
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
                    "Normalized daily mean activity per recording with method {}.\n".format(
                        options["method"]
                    )
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

    def plot_separate_distances(self, data, options, infos):
        plt_df = data["df"]
        res = []
        y_range = max(plt_df.trend_norm) - min(plt_df.trend_norm)
        ylims = [
            min(plt_df.trend_norm) - 0.1 * y_range,
            max(plt_df.trend_norm) + 0.1 * y_range,
        ]
        plt_gt_norm = (
            ggplot(
                data=plt_df.loc[plt_df.type == "ground_truth"],
                mapping=aes("date", "trend_norm", color="type"),
            )
            + geom_line(color="blue")
            + ggtitle(
                "Normalized daily mean activity per recording. \n"
                + " Euclidean distance to reference: {}".format(data["distance_norm"])
            )
            + xlab("Date")
            + ylab("Normalized daily mean activity per recording")
            + ylim(ylims)
            + scale_color_discrete(labels=["Reference"])
            + scale_x_datetime(labels=format_date_short)
            + theme_classic()
            + theme(axis_text_x=element_text(angle=45))
        )

        plt_norm = (
            ggplot(
                data=plt_df.loc[plt_df.type == "DLBD"],
                mapping=aes("date", "trend_norm", color=["red"]),
            )
            + geom_line()
            + ylim(ylims)
            + scale_color_discrete(labels=["Model"], color=["red"])
            + theme_classic()
            + theme(
                axis_title=element_blank(),
                axis_ticks_major=element_blank(),
                axis_text=element_blank(),
            )
        )
        res.append(plt_gt_norm)
        res.append(plt_norm)

        return res

    def evaluate(self, data, options, infos):
        if not self.check_database(data, options, infos):
            return {}
        # if infos["database"] not in options.get("phenology_databases", []):
        #     common_utils.print_info(
        #         (
        #             "Database {} is not part of the accepted databases for the 'Phenology' "
        #             + "evaluator described in the 'phenology_databases' option. Skipping."
        #         ).format(options["scenario_info"]["database"])
        #     )
        #     return {}
        method = options["method"]
        stats = EVALUATORS[method].evaluate(data, options, infos)

        matches = stats["matches"]

        daily_tags_duration = self.get_daily_activity(
            data[1]["tags_df"], options, "tag"
        )
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
            plts = self.draw_plots(
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
                infos=infos,
            )
            if "plots" in stats:
                stats["plots"].update(plts)
            else:
                stats["plots"] = plts

        return stats
