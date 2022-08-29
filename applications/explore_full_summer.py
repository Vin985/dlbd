#%%
from pathlib import Path

import pandas as pd
from dlbd.applications.phenology.phenology_evaluator import PhenologyEvaluator
from dlbd.data.tag_utils import filter_classes, flatten_tags
from dlbd.evaluation import EVALUATORS
from pandas_path import path

from plotnine import (
    aes,
    annotate,
    element_blank,
    geom_line,
    geom_smooth,
    ggplot,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_classic,
    xlab,
    ylab,
    ylim,
)


tag_path = "C:/UMoncton/Doctorat/data/dl_training/datasets/full_summer1/original_mel32_512_None_None/test_tags_df_biotic.feather"
tags_df = pd.read_feather(tag_path)
tags_df = tags_df.rename(columns={"file_name": "recording_id"})
tags_df["tag_duration"] = tags_df["tag_end"] - tags_df["tag_start"]

#%%
classes_df = pd.read_csv(
    "C:/UMoncton/Doctorat/dev/dlbd/config/classes.csv", skip_blank_lines=True
)


classes = (
    classes_df.loc[
        classes_df["class_type"] == "biotic"  # pylint: disable=unsubscriptable-object
    ]
    .tag.str.lower()
    .values
)

opts = {"classes": classes, "tags": {}}

filtered_df = filter_classes(tags_df, opts)


#%%

EVALUATORS.register_evaluator(PhenologyEvaluator)

print(filtered_df.tag_duration.sum())

opts = {"method": "citynet", "activity_threshold": 0.75}

daily_activity = EVALUATORS["phenology"].get_daily_trends(filtered_df, opts, "tag")
print(daily_activity)
df = daily_activity["trends_df"]

df.plot("date", "trend")

#%%

filters = ["biotic", "shorebird", "goose", "loon", "passerine", "unkn"]
res = []
trend_sum = None

for filt in filters:
    if filt == "biotic":
        classes = (
            classes_df.loc[
                classes_df["class_type"]
                == "biotic"  # pylint: disable=unsubscriptable-object
            ]
            .tag.str.lower()
            .values
        )
    else:
        classes = [filt]
    filtered_df = filter_classes(tags_df, {"classes": classes, "tags": {}})
    daily_activity = EVALUATORS["phenology"].get_daily_trends(
        filtered_df, opts, "tag"
    )

    df = daily_activity["trends_df"]
    df["type"] = filt
    res.append(df)
    if not filt == "biotic":
        if trend_sum is None:
            trend_sum = df[["date", "trend"]]
        else:
            trend_sum.loc[:, "trend"] = trend_sum["trend"] + df["trend"]

all_filters = pd.concat(res)
trend_sum.loc[:, "type"] = "sum"

all_filters = pd.concat([all_filters, trend_sum])

all_plt = (
    ggplot(data=all_filters, mapping=aes(x="date", y="trend", color="type"))
    + geom_line()
)



print(all_plt)

#%%
