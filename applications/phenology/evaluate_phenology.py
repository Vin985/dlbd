#%%

import ast
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mouffet.utils.file as file_utils
import numpy as np
import pandas as pd
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.data.tag_utils import flatten_tags
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)
from fastdtw import fastdtw
from mouffet.training.training_handler import TrainingHandler
from mouffet.utils.file import ensure_path_exists
from numpy.core.fromnumeric import std
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
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from statsmodels.tsa.seasonal import seasonal_decompose


def load_model_options(opts, updates):
    model_opt = ast.literal_eval(opts)
    model_opt.update(updates)
    return model_opt


def get_file_singing_duration(df):
    flattened = flatten_tags(df)
    return flattened.tag_duration.sum()


def get_reference_tags_duration(dest_path, evaluator, overwrite=False):
    if not dest_path.exists() or overwrite:
        evaluator.data_handler.check_datasets()

        tags = evaluator.data_handler.load_dataset(
            evaluator.data_handler.get_database_options("full_summer1"),
            "test",
            load_opts={"file_types": "tags_df"},
        )["tags_df"]

        res_df = (
            tags.groupby("file_name")
            .apply(get_file_singing_duration)
            .reset_index(name="duration")
        )

        res_df[
            ["site", "plot", "date", "time", "to_drop"]
        ] = res_df.file_name.str.split("_", expand=True)
        res_df = res_df.assign(
            full_date=[str(x) + "_" + y for x, y in zip(res_df["date"], res_df["time"])]
        )
        res_df["full_date"] = pd.to_datetime(
            res_df["full_date"], format="%Y-%m-%d_%H%M%S"
        )
        res_df["date"] = pd.to_datetime(res_df["date"], format="%Y-%m-%d")

        file_utils.ensure_path_exists(dest_path, is_file=True)
        res_df.to_feather(dest_path)

    else:
        res_df = pd.read_feather(dest_path)
    return res_df


def check_models(config, model_opts):
    # * Get reference
    models = config.get("models", [])
    if not models:
        models_dir = model_opts.get("model_dir")
        models_stats_path = Path(models_dir / TrainingHandler.MODELS_STATS_FILE_NAME)
        models_stats = None
        if models_stats_path.exists():
            models_stats = pd.read_csv(models_stats_path).drop_duplicates(
                "opts", keep="last"
            )
        if models_stats is not None:
            model_ids = config.get("model_ids", [])
            if model_ids:
                models_stats = models_stats.loc[models_stats.model_id.isin(model_ids)]
            models = [
                load_model_options(row.opts, model_opts)
                for row in models_stats.itertuples()
            ]
            # config["models"] = [models[0]]
            config["models"] = models  # [0:2]

    return config


#%%

wavs_dir = "wavs"
tags_file = "tags.csv"
models_dir = Path("/home/vin/Desktop/results/candidates_models")
flattened_tags_path = Path(
    "/home/vin/Doctorat/dev/dlbd/applications/phenology/results/merged_tags.feather"
)

evaluation_config_path = (
    "/home/vin/Doctorat/dev/dlbd/applications/phenology/evaluation_config.yaml"
)

evaluation_config = file_utils.load_config(evaluation_config_path)


model_opts = {"model_dir": models_dir}

check_models(evaluation_config, model_opts)

sd_evaluator = SongDetectorEvaluationHandler(
    opts=evaluation_config, dh_class=AudioDataHandler
)

ref_df = get_reference_tags_duration(flattened_tags_path, sd_evaluator, overwrite=True)


ref_per_day = ref_df[["date", "duration"]].groupby("date").mean().reset_index()
ref_per_day["type"] = "ground_truth"
ref_per_day["mov_avg"] = ref_per_day.duration.rolling(4, center=True).mean()
ref_per_day["diff"] = 0

#%%

evaluator = SongDetectorEvaluationHandler(
    opts=evaluation_config, dh_class=AudioDataHandler
)


#%%


def get_file_events_duration(df):
    if df.empty:
        return 0
    return df.drop_duplicates("event_id")["event_duration"].sum()


def get_daily_events_duration(df):
    return df.drop_duplicates("event_id")["event_duration"].mean()


agg_path = Path(evaluator.opts["predictions_dir"]) / "matches_per_day.feather"
full_matches_path = Path(evaluator.opts["predictions_dir"]) / "matches_per_file.feather"
full_stats_path = Path(evaluator.opts["predictions_dir"]) / "full_stats.pickle"

full_matches = []

if agg_path.exists():
    res_df = pd.read_feather(agg_path)
else:
    res = []
    if full_stats_path.exists():
        stats = pickle.load(open(full_stats_path, "rb"))
    else:
        stats = evaluator.evaluate()
        with open(ensure_path_exists(full_stats_path, is_file=True), "wb") as f:
            pickle.dump(stats, f, -1)
            print("Saved file: ", full_stats_path)
    for i, model_stats in enumerate(stats):
        matches = model_stats["matches"]
        # * Get total duration per file
        matches[["file_name", "event_id", "event_duration"]].groupby("file_name").apply(
            get_file_events_duration
        ).reset_index()
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
        full_matches.append(matches)

        # * Get mean duration per day
        matches_per_day = (
            matches.groupby("date")
            .apply(get_daily_events_duration)
            .reset_index(name="duration")
        )
        matches_per_day["type"] = stats[i]["stats"].model.iloc[0]
        matches_per_day["mov_avg"] = matches_per_day.duration.rolling(
            4, center=True
        ).mean()
        matches_per_day["diff"] = matches_per_day["mov_avg"] - ref_per_day["mov_avg"]
        res.append(matches_per_day)

    res_df = pd.concat([ref_per_day] + res).reset_index()
    res_df.to_feather(agg_path)

    full_matches_df = pd.concat(full_matches).reset_index()
    full_matches_df.to_feather(full_matches_path)
print(res_df)
#%%


plots = []
norm_plots = []
ref = None

for model in res_df["type"].unique():
    tmp = res_df.loc[res_df.type == model]
    tmp_df = tmp[["date", "duration"]].set_index("date").fillna(0)
    result_mul = seasonal_decompose(tmp_df, model="additive", extrapolate_trend="freq")
    trend = result_mul.trend.reset_index(name="duration")
    trend["norm"] = (trend.duration - trend.duration.mean()) / trend.duration.std()
    if model == "ground_truth":
        ref = trend
    elif ref is not None:
        dtw_distance, warp_path = fastdtw(ref.norm, trend.norm)
        eucl_distance = euclidean(ref.norm, trend.norm)

        print(
            "Distances for model {}: dtw: {}; eucl: {}".format(
                model, dtw_distance, eucl_distance
            )
        )
    peaks = find_peaks(trend["norm"])[0]
    peaks_df = trend.iloc[peaks]
    print(peaks_df)
    tmp_plt = (
        ggplot(
            data=tmp.dropna(),
            mapping=aes(
                "date",
                "duration",
            ),
        )
        + geom_point()
        + geom_line(mapping=aes("date", "mov_avg"), colour="#ff0000")
        + geom_line(
            data=trend,
            mapping=aes("date", "duration"),
            colour="#0000ff",
        )
    )

    norm_plt = (
        ggplot(
            data=trend.dropna(),
            mapping=aes(
                "date",
                "norm",
            ),
        )
        + geom_line(colour="#ff0000")
        + geom_point(data=peaks_df, mapping=aes("date", "norm"), colour="#00ff00")
        + geom_line(
            data=ref,
            mapping=aes("date", "norm"),
            colour="#0000ff",
        )
        + ggtitle(model)
    )
    plots.append(tmp_plt)
    norm_plots.append(norm_plt)

cur_time = datetime.now()
prefix = cur_time.strftime("%H%M%S")

save_as_pdf_pages(
    plots,
    ensure_path_exists(
        Path(evaluator.opts["evaluation_dir"])
        / cur_time.strftime("%Y%m%d")
        / (prefix + "_phenology_all.pdf"),
        is_file=True,
    ),
)

save_as_pdf_pages(
    norm_plots,
    ensure_path_exists(
        Path(evaluator.opts["evaluation_dir"])
        / cur_time.strftime("%Y%m%d")
        / (prefix + "_phenology_norm_all.pdf"),
        is_file=True,
    ),
)

#%%

# * TIME SERIES ANALYSIS


gt_df = res_df.loc[
    res_df.type == "DLBDL_noes_lr-0.01_epochs-50", ["date", "duration"]
].set_index("date")
gt_df
#%%


result_mul = seasonal_decompose(gt_df, model="multiplicative", extrapolate_trend="freq")

# Additive Decomposition
result_add = seasonal_decompose(gt_df, model="additive", extrapolate_trend="freq")

# Plot
plt.rcParams.update({"figure.figsize": (10, 10)})
result_mul.plot().suptitle("Multiplicative Decompose", fontsize=22)
result_add.plot().suptitle("Additive Decompose", fontsize=22)
plt.show()


#%%

# import numpy as np
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean


# for model in res_df["type"].unique():
#     if model == "ground_truth":
#         continue

#     dtw_distance, warp_path = fastdtw(
#         res_df.duration.loc[res_df["type"] == "ground_truth"],
#         res_df.duration.loc[res_df["type"] == model].fillna(0),
#         dist=euclidean,
#     )

#     print("Distance for model {}: {}".format(model, dtw_distance))

#%%

from pmdarima.arima import ADFTest, auto_arima

adf = ADFTest(alpha=0.05)

# gt = res_df.duration.loc[res_df["type"] == "ground_truth"]
gt = (
    res_df.loc[res_df.type == "ground_truth", ["date", "duration"]]
    .set_index("date")
    .fillna(0)
)
print(adf.should_diff(gt.duration))
gt["durI"] = gt["duration"] - gt["duration"].shift(1)
gt = gt.dropna()
print(adf.should_diff(gt.durI))

mod1 = (
    res_df.loc[res_df.type == "DLBDL_noes_lr-0.01_epochs-50", ["date", "duration"]]
    .set_index("date")
    .fillna(0)
)
adf.should_diff(mod1.duration)
mod1["durI"] = mod1["duration"] - mod1["duration"].shift(1)
mod1 = mod1.dropna()
adf.should_diff(mod1.durI)
#%%

# result_mul = seasonal_decompose(
#     gt_df2, model="multiplicative", period=200, extrapolate_trend="freq"
# )

# # Additive Decomposition
# result_add = seasonal_decompose(
#     gt_df2, period=200, model="additive", extrapolate_trend="freq"
# )

# # Plot
# plt.rcParams.update({"figure.figsize": (10, 10)})
# result_mul.plot().suptitle("Multiplicative Decompose", fontsize=22)
# result_add.plot().suptitle("Additive Decompose", fontsize=22)
# plt.show()

#%%
from pandas.plotting import autocorrelation_plot

# autocorrelation_plot(gt_df)

deseasonalized = gt_df.duration.values / result_mul.seasonal

# Plot
plt.plot(deseasonalized)
plt.title("Drug Sales Deseasonalized", fontsize=16)
plt.plot()


#%%

gt_norm = (
    res_df.loc[res_df.type == "ground_truth", ["date", "duration"]]
    .set_index("date")
    .fillna(0)
)


gt_norm["norm"] = (gt_norm.duration - gt_norm.duration.mean()) / gt_norm.duration.std()


gt_norm.plot(y="duration")
gt_norm.plot(y="norm")
