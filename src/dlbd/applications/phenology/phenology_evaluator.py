from datetime import datetime

import pandas as pd
from mouffet import common_utils
from mouffet.evaluation import Evaluator
from pandas_path import path  # pylint: disable=unused-import
from plotnine import *
from scipy.spatial.distance import euclidean

from ...evaluation import EVALUATORS
from ...data.tag_utils import flatten_tags
from ...plots.distance import plot_distances


class PhenologyEvaluator(Evaluator):

    NAME = "phenology"

    PLOTS = {"distances": plot_distances}

    def requires(self, options):
        return EVALUATORS[options["method"]].requires(options)

    def file_event_duration(self, df, method):
        return EVALUATORS[method].file_event_duration(df)

    def file_positive_event_duration(self, df, method):
        return EVALUATORS[method].file_positive_event_duration(df)

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

    def get_rolling_trends(self, df, options, df_type, default_period=7):
        res = {}
        tmp = df.copy()
        roll = tmp["total_duration"].rolling(
            options.get("period", default_period), center=True
        )

        tmp.loc[:, "trend"] = roll.mean()
        tmp.loc[:, "trend_std"] = roll.std()

        tmp = tmp.dropna()

        tmp.loc[:, "trend_norm"] = (tmp.trend - tmp.trend.mean()) / tmp.trend.std()

        if df_type == "tag":
            tmp.loc[:, "type"] = "ground_truth"
        elif options.get("scenario_info"):
            tmp.loc[:, "type"] = options["scenario_info"]["model"]
        else:
            tmp.loc[:, "type"] = options.get("plot", "unknown_plot")

        res["trends_df"] = tmp.reset_index()

        return res

    def get_daily_trends(self, df, options, df_type="event"):
        res = {}
        # * Get total duration per file
        if df_type == "event" and options.get("only_positives", False):
            df_type = "positive_" + df_type

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

        agg_method = options.get("daily_aggregation", "sum")

        by_hour = (
            file_song_duration[["date_hour", "total_duration"]]
            .groupby("date_hour")
            .agg(agg_method)
        )
        by_hour = by_hour.asfreq("H", fill_value=0)

        daily_duration = by_hour.resample("D").agg(agg_method)
        daily_duration.index.name = "date"

        trends = self.get_rolling_trends(daily_duration, options, df_type)

        res.update(trends)

        return res

    def get_ENAB_final_trends(self, data, options, df_type):
        res = {}
        df = data.copy()
        df[["recording", "segment"]] = df.recording_id.path.stem.str.split(
            "_", expand=True
        )[[1, 3]]
        df = df.loc[df.recording == "1"]
        df.segment = df.segment.astype(int)

        if df_type == "tag" and options.get("remove_crows", False):
            df = df.loc[df.tag != "AMCR"]

        file_song_duration = (
            df.groupby(["segment"])
            .apply(
                getattr(self, "file_" + df_type + "_duration"), method=options["method"]
            )
            .reset_index()
            .rename(columns={0: "total_duration"})
        )
        fsd = file_song_duration.copy().set_index(file_song_duration.segment)
        fsd = fsd.reindex(list(range(1, 37)), fill_value=0)
        fsd.loc[:, "date"] = pd.date_range(start=0, periods=36, freq="5min")
        fsd = fsd.set_index("date")

        res["file_song_duration"] = fsd

        trends = self.get_rolling_trends(fsd, options, df_type, default_period=5)

        res.update(trends)

        return res

    def get_trends(self, data, infos, options, df_type):
        db = infos["database"]
        activity_func_name = "get_" + db + "_trends"
        if not hasattr(self, activity_func_name):
            activity_func_name = "get_daily_trends"
        return getattr(self, activity_func_name)(data, options, df_type)

    # def plot_distances(self, data, options, infos):
    #     plt_df = data["df"]
    #     res = []
    #     if options.get("plot_real_distance", True):
    #         tmp_plt = (
    #             ggplot(
    #                 data=plt_df,
    #                 mapping=aes("date", "trend", color="type"),
    #             )
    #             + geom_line()
    #             + ggtitle(
    #                 "Daily mean activity per recording with method {}.\n".format(
    #                     options["method"]
    #                 )
    #                 + " Euclidean distance to reference: {}".format(data["distance"])
    #             )
    #             + xlab("Date")
    #             + ylab("Daily mean activity per recording (s)")
    #             + scale_color_discrete(labels=["Reference", "Model"])
    #             + scale_x_datetime(labels=format_date_short)
    #             + theme_classic()
    #             + theme(axis_text_x=element_text(angle=45))
    #         )
    #         if options.get("add_points", False):
    #             tmp_plt = tmp_plt + geom_point(mapping=aes(y="total_duration"))
    #         res.append(tmp_plt)
    #     if options.get("plot_norm_distance", True):
    #         tmp_plt_norm = (
    #             ggplot(
    #                 data=plt_df,
    #                 mapping=aes("date", "trend_norm", color="type"),
    #             )
    #             + geom_line()
    #             + ggtitle(
    #                 "Normalized daily mean activity per recording with method {}.\n".format(
    #                     options["method"]
    #                 )
    #                 + " Euclidean distance to reference: {}".format(
    #                     data["distance_norm"]
    #                 )
    #             )
    #             + xlab("Date")
    #             + ylab("Normalized daily mean activity per recording")
    #             + scale_color_discrete(labels=["Model", "Reference"])
    #             + scale_x_datetime(labels=format_date_short)
    #             + theme_classic()
    #             + theme(axis_text_x=element_text(angle=45))
    #         )
    #         res.append(tmp_plt_norm)

    #     return res

    def evaluate(self, data, options, infos):
        if not self.check_database(data, options, infos):
            return {}
        method = options["method"]
        stats = EVALUATORS[method].evaluate(data, options, infos)

        matches = stats["matches"]

        tags_trends = self.get_trends(data[1]["tags_df"], infos, options, "tag")
        events_trends = self.get_trends(matches, infos, options, "event")

        trends = pd.concat(
            [
                tags_trends["trends_df"],
                events_trends["trends_df"],
            ]
        )
        trends["type"] = trends["type"].astype("category")
        trends["type"] = trends["type"].cat.reorder_categories(
            ["ground_truth", options["scenario_info"]["model"]]
        )

        eucl_distance = round(
            euclidean(
                tags_trends["trends_df"].trend,
                events_trends["trends_df"].trend,
            ),
            3,
        )
        eucl_distance_norm = round(
            euclidean(
                tags_trends["trends_df"].trend_norm,
                events_trends["trends_df"].trend_norm,
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
        stats["trends_df"] = {"trends_df": trends}
        # stats["tags_trends"] = tags_trends
        # stats["events_duration"] = events_trends

        if options.get("draw_plots", False):
            plts = self.draw_plots(
                data={
                    "df": trends,
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
