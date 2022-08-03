#%%
import pandas as pd
from dlbd.applications.phenology.phenology_evaluator import PhenologyEvaluator
from dlbd.data.tag_utils import filter_classes, flatten_tags
from dlbd.evaluation import EVALUATORS
from pandas_path import path

tag_path = "/mnt/win/UMoncton/Doctorat/data/dl_training/datasets/full_summer1//original_mel32_512_512_None/test_tags_df_biotic.feather"


tags_df = pd.read_feather(tag_path)
tags_df = tags_df.rename(columns={"file_name": "recording_id"})
tags_df["tag_duration"] = tags_df["tag_end"] - tags_df["tag_start"]

#%%
classes_df = pd.read_csv(
    "/mnt/win/UMoncton/Doctorat/dev/dlbd/config/classes.csv", skip_blank_lines=True
)


classes = (
    classes_df.loc[
        classes_df["class_type"] == "biotic"  # pylint: disable=unsubscriptable-object
    ]
    .tag.str.lower()
    .values
)

# classes = ["unkn"]

opts = {"classes": classes, "tags": {}}

tags_df = filter_classes(tags_df, opts)

#%%


def extract_recording_info(df):
    df[
        ["site", "plot", "date", "time", "to_drop"]
    ] = df.recording_id.path.stem.str.split("_", expand=True)
    df = df.assign(full_date=[str(x) + "_" + y for x, y in zip(df["date"], df["time"])])
    df["full_date"] = pd.to_datetime(df["full_date"], format="%Y-%m-%d_%H%M%S")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.drop(columns=["to_drop"])
    return df


#%%

# flattened = (
#     tags_df.groupby("recording_id")
#     .apply(flatten_tags)
#     .reset_index()
#     .drop(columns="level_1")
# )

# flat = extract_recording_info(flattened)


#%%


EVALUATORS.register_evaluator(PhenologyEvaluator)


opts = {"method": "citynet"}

daily_activity = EVALUATORS["phenology"].get_daily_activity(tags_df, opts, "tag")
df = daily_activity["daily_duration"]

df.plot("date", "trend")
