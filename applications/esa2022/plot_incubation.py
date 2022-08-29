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

plot_name = "BARW_0"
overlap = 0.75

dest_root_dir = Path("/mnt/win/UMoncton/Doctorat/dev/dlbd/applications/esa2022/results")
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
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/sites_deployment_2018.xlsx"
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
    daily_activity = EVALUATORS["phenology"].get_daily_activity(
        events_df, opts, "event"
    )
    df = daily_activity["daily_duration"]

    df.plot("date", "trend")

#%%


def label_x(dates):
    res = [
        (datetime.datetime(2018, 1, 1) + datetime.timedelta(x)).strftime("%d-%m")
        for x in dates
    ]
    print(res)
    return res


def create_file_name(type, site, prefix, suffix):
    return "_".join(filter(None, [prefix, type, site, suffix]))


def fill_missing_days(df, start, end):
    tmp = df.set_index("julian")
    new_index = pd.Index(np.arange(start, end), name="julian")
    tmp = tmp.set_index("julian").reindex(new_index).reset_index()


def fill_missing_range(df, field, range_from, range_to, range_step=1, fill_with=0):
    return (
        df.merge(
            how="right",
            on=field,
            right=pd.DataFrame({field: np.arange(range_from, range_to, range_step)}),
        )
        .sort_values(by=field)
        .reset_index()
        .fillna(fill_with)
        .drop(["index", "level_0"], axis=1)
    )


def get_nesting_stages(df_nest, stages=["incubation", "hatch"]):
    res = {}

    for stage in stages:
        tmp_df = (
            df_nest.loc[df_nest.type == stage]
            .groupby(["julian", "type"], as_index=False)
            .uniqueID.count()
            .reset_index()
            .copy()
        )

        start = tmp_df.julian.min()
        end = tmp_df.julian.max()
        lbl_pos = start + (end - start) / 2
        df = fill_missing_range(tmp_df, "julian", start, end)
        df.type.fillna(stage)

        res[stage] = {
            "start": start,
            "end": end,
            "lbl_pos": lbl_pos,
            "data": df,
            "max": df.uniqueID.max(),
        }

    return res


DEFAULT_LABELS = {
    "songs": {
        "xlab": "Day",
        "ylab": "Mean daily events detected",
        "periods": ["Incubation initiation", "Hatching"],
    },
    "nesting": {
        "xlab": "Day",
        "ylab": "Number of nest initiation/hatch",
    },
}


def incubation_plot(
    data,
    site,
    nest_data_path,
    save=True,
    ext=".png",
    prefix="",
    suffix="",
    dest_dir="",
    y="trend",
    labels={},
):
    # df = data.loc[data.site == site]

    data["julian"] = data.date.dt.dayofyear
    lbls = deepcopy(DEFAULT_LABELS)
    lbls.update(labels)
    df_nest = pd.read_excel(nest_data_path)
    df_nest["julian"] = df_nest.date.dt.dayofyear
    df_nest = df_nest.loc[
        (df_nest["plot"] == "brw0"),
    ]

    stages = get_nesting_stages(df_nest)

    xmin = min(stages["incubation"]["start"], data.julian.min())
    xmax = min(stages["hatch"]["end"] + 2, data.julian.max() + 2)

    yrange = data[y].max() - data[y].min()
    bufmin = 0.05 * yrange
    bufmax = 0.1 * yrange
    ymin = 0  # data["total_duration"].min()  # - bufmin
    ymax = 300  # data["total_duration"].max()  # + bufmax

    se_plot = (
        ggplot(data=data, mapping=aes(x="julian", y=y, colour="site"))
        + xlab(lbls["songs"]["xlab"])
        + ylab(lbls["songs"]["ylab"])
        + geom_point(aes(y="total_duration"))
        + geom_line()
        + theme_classic()
        + theme(legend_position="none")
        + annotate(
            "rect",
            xmin=[stages["incubation"]["start"], stages["hatch"]["start"]],
            xmax=[stages["incubation"]["end"], min(stages["hatch"]["end"], xmax)],
            ymin=-math.inf,
            ymax=math.inf,
            alpha=0.1,
            fill=["red", "blue"],
        )
        + annotate(
            "text",
            x=[stages["incubation"]["lbl_pos"], stages["hatch"]["lbl_pos"]],
            y=data[y].max() + 0.1 * yrange,
            label=lbls["songs"]["periods"],
        )
        + scale_x_continuous(labels=label_x, limits=[xmin, xmax])
        + ylim(ymin, ymax)
    )

    inc_plot = (
        ggplot(data=stages["incubation"]["data"], mapping=aes(x="julian", y="uniqueID"))
        + xlab(lbls["nesting"]["xlab"])
        + ylab(lbls["nesting"]["ylab"])
        + theme_classic()
        + theme(
            axis_line=element_blank(),
            axis_title_y=element_blank(),
            axis_ticks_major_y=element_blank(),
            axis_text_y=element_blank(),
        )
        # + geom_point()
        # + geom_line(linetype="dashed",)
        + geom_smooth(
            linetype="dashed",
            method="mavg",
            # color="red",
            se=False,
            method_args={
                "window": 3,
                "center": True,
                "min_periods": 1,
                "win_type": "triang",
            },
        )
        # + geom_point(data=stages["hatch"]["data"])
        # + geom_line(data=stages["hatch"]["data"], linetype="dotted",)
        + geom_smooth(
            data=stages["hatch"]["data"],
            linetype="dotted",
            method="mavg",
            se=False,
            # color="blue",
            method_args={
                "window": 3,
                "center": True,
                "min_periods": 1,
                "win_type": "triang",
            },
        )
        + scale_x_continuous(labels=label_x, limits=[xmin, xmax])
        + scale_y_continuous(
            position="right",
            limits=[0, max(stages["incubation"]["max"], stages["hatch"]["max"]) * 1.2],
        )
    )

    if save:
        se_name = create_file_name("sound_events", site, prefix, suffix)
        se_plot.save(
            file_utils.ensure_path_exists(dest_dir / (se_name + ext), is_file=True),
            height=8,
            width=8,
            dpi=150,
        ),
        # inc_name = create_file_name("incubation", site, prefix, suffix)
        # inc_plot.save(
        #     file_utils.ensure_path_exists(dest_dir / (inc_name + ext), is_file=True),
        #     height=8,
        #     width=8,
        #     dpi=150,
        # )
    return (se_plot, inc_plot)


#%%


(barw_se, barw_inc) = incubation_plot(
    df,
    plot_name,
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Nest Monitoring/2018/BARW/BARW_nest_2018.xlsx",
    # save=False,
    prefix=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_esa2022_"
    + events_file_prefix,
    dest_dir=plots_dest_dir,
)
