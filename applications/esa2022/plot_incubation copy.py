#%%

import datetime
import math
from copy import deepcopy
from pathlib import Path

import mouffet.utils.file as file_utils
import numpy as np
import pandas as pd
from dlbd.applications.phenology import PhenologyEvaluator
from dlbd.evaluation import EVALUATORS
from plotnine import (
    aes,
    annotate,
    element_blank,
    geom_line,
    geom_point,
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


EVALUATORS.register_evaluator(PhenologyEvaluator)

plot_name = "IGLO_B"
overlap = 0.75

dest_root_dir = Path("C:/UMoncton/Doctorat/dev/dlbd/applications/esa2022/results")
events_dest_dir = dest_root_dir / "events"
plots_dest_dir = dest_root_dir / "plots"

predictions_path = (
    dest_root_dir
    / "predictions"
    / plot_name
    / "predictions_overlap{}.feather".format(overlap)
)
preds = pd.read_feather(predictions_path)


preds = preds.rename(columns={"recording_path": "recording_id"})

#%%


#%%

site_data = pd.read_excel(
    "C:/UMoncton/OneDrive - Université de Moncton/Data/sites_deployment_2018.xlsx"
)


opts = {
    "method": "standard",
    "activity_threshold": 0.92,
    "min_duration": 0.2,
    "end_threshold": 0.5,
    "gtc_threshold": 0,
    "dtc_threshold": 0,
    "recording_info_type": "audiomoth2018",
    "depl_start": site_data.loc[site_data["plot"] == plot_name, "depl_start"].iloc[0],
    "depl_end": site_data.loc[site_data["plot"] == plot_name, "depl_end"].iloc[0],
    "period": 7,
}


events_file_prefix = "{}_{}_period{}".format(
    opts["method"], opts["activity_threshold"], opts.get("period", 7)
)
events_path = events_dest_dir / ("event_" + events_file_prefix + ".feather")


events_df = EVALUATORS[opts["method"]].filter_predictions(preds, opts)
events_df.to_feather(file_utils.ensure_path_exists(events_path, is_file=True))


if not events_df.empty:
    daily_activity = EVALUATORS["phenology"].get_daily_trends(
        events_df, opts, "event"
    )
    df = daily_activity["trends_df"]

    df.plot("date", "trend")
from plotnine import ggplot, geom_line, aes

all_plt = (
    ggplot(data=df, mapping=aes(x="date", y="trend", color="type"))
    + geom_line()
)
print(all_plt)

#%%

