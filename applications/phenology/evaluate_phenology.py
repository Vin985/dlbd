#%%

import ast
from pathlib import Path

import mouffet.utils.file as file_utils
import pandas as pd
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.data.tag_utils import flatten_tags
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)
from mouffet.training.training_handler import TrainingHandler
from plotnine import (
    aes,
    geom_line,
    geom_point,
    geom_smooth,
    ggplot,
    facet_grid,
    save_as_pdf_pages,
)


def load_model_options(opts, updates):
    model_opt = ast.literal_eval(opts)
    model_opt.update(updates)
    return model_opt


def get_flattened_reference_tags(flattened_path, evaluator, overwrite=False):
    if not flattened_path.exists() or overwrite:
        evaluator.data_handler.check_datasets(
            # db_types=["training", "validation", "test"]
        )

        tags = evaluator.data_handler.load_dataset(
            evaluator.data_handler.get_database_options("full_summer1"),
            "test",
            load_opts={"file_types": "tags_df"},
        )["tags_df"]

        flattened = (
            tags.groupby("file_name")
            .apply(flatten_tags)
            .reset_index()
            .rename(columns={"level_1": "tag_index", "tag_duration": "duration"})
        )

        flattened[
            ["site", "plot", "date", "time", "to_drop"]
        ] = flattened.file_name.str.split("_", expand=True)
        flattened["date"] = pd.to_datetime(flattened["date"], format="%Y-%m-%d")

        file_utils.ensure_path_exists(flattened_path, is_file=True)
        flattened.to_feather(flattened_path)

    else:
        flattened = pd.read_feather(flattened_path)
    return flattened


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

ref_df = get_flattened_reference_tags(
    flattened_tags_path, sd_evaluator, overwrite=False
)
ref_per_day = ref_df[["date", "duration"]].groupby("date").mean().reset_index()
ref_per_day["type"] = "ground_truth"
ref_per_day["mov_avg"] = ref_per_day.duration.rolling(4, center=True).mean()
ref_per_day["diff"] = 0

#%%

evaluator = SongDetectorEvaluationHandler(
    opts=evaluation_config, dh_class=AudioDataHandler
)


#%%


def get_events_duration(df):
    return df.drop_duplicates("event_id")["event_duration"].mean()


agg_path = Path(evaluator.opts["predictions_dir"]) / "matches_per_day.feather"

if agg_path.exists():
    res_df = pd.read_feather(agg_path)
else:
    res = []
    stats = evaluator.evaluate()
    for i, model_stats in enumerate(stats):
        matches = model_stats["matches"]
        matches[
            ["site", "plot", "date", "time", "to_drop"]
        ] = matches.file_name.str.split("_", expand=True)
        matches["date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d")
        matches_per_day = (
            matches.groupby("date")
            .apply(get_events_duration)
            .reset_index()
            .rename(columns={0: "duration"})
        )
        matches_per_day["type"] = stats[i]["stats"].model.iloc[0]
        matches_per_day["mov_avg"] = matches_per_day.duration.rolling(
            4, center=True
        ).mean()
        matches_per_day["diff"] = matches_per_day["mov_avg"] - ref_per_day["mov_avg"]
        res.append(matches_per_day)

    res_df = pd.concat([ref_per_day] + res).reset_index()
    res_df.to_feather(agg_path)
print(res_df)
#%%


plots = []

for model in res_df["type"].unique():
    tmp = res_df.loc[res_df.type == model]
    plt = (
        ggplot(
            data=tmp.dropna(),
            mapping=aes("date", "duration", color="factor(type)"),
        )
        + geom_point()
        + geom_line(mapping=aes("date", "mov_avg", color="factor(type)"))
    )
    plots.append(plt)

save_as_pdf_pages(plots, Path(evaluator.opts["evaluation_dir"]) / "phenology_all.pdf")
