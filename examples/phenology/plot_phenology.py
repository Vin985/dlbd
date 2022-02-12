#%%
from pathlib import Path
import pandas as pd

from dlbd.applications.phenology import PhenologyEvaluator
from dlbd.evaluation import EVALUATORS
from pandas_path import path  # pylint: disable=unused-import
from statsmodels.tsa.seasonal import seasonal_decompose

from plotnine import aes, geom_line, ggplot, ggtitle, save_as_pdf_pages

EVALUATORS.register_evaluator("phenology", PhenologyEvaluator)
preds_root = Path("/home/vin/Doctorat/dev/dlbd/results/predict")

plots = [
    {
        "src_path": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT12_EC20",
        "name": "LYT12_EC20",
    },
    {
        "src_path": "/media/vin/BigMama/Sylvain/AL57",
        "name": "2021_BARW_0_AL57",
    },
    {
        "src_path": "/media/vin/BigMama/Sylvain/AL58",
        "name": "2021_BARW_8_AL58",
    },
]

for plot in plots:

    preds = pd.read_feather(preds_root / plot["name"] / "predictions.feather")
    preds = preds.rename(columns={"recording_path": "recording_id"})

    opts = {
        "event_threshold": 0.5,
        # "method": "citynet",
        "method": "standard",
        "activity_threshold": 0.9,
        "min_duration": 0.1,
        "end_threshold": 0.3,
        "recording_info_type": "audiomoth2019",
    }

    events = EVALUATORS[opts["method"]].get_events(preds, None, opts)
    if not events.empty:
        daily_activity = EVALUATORS["phenology"].get_daily_activity(
            events, opts, "event"
        )

        df = daily_activity["daily_duration"]
        plt = (
            ggplot(
                data=df,
                mapping=aes("date", "trend", color="type"),
            )
            + geom_line()
        )

        plt_norm = (
            ggplot(
                data=df,
                mapping=aes("date", "trend_norm", color="type"),
            )
            + geom_line()
        )

        save_as_pdf_pages(
            [plt, plt_norm],
            preds_root / plot["name"] + "_" + opts["method"] + "_plots.pdf",
        )
    else:
        print("No events detected for {}".format(plot["name"]))
