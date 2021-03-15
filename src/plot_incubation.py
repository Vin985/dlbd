#%%

import datetime
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
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

from examples.cjcc.events_plot import EventsPlot

src_dir = Path("/mnt/win/UMoncton/Doctorat/dev/dlbd/results/predictions/CJCC/")
dest_dir = Path("/mnt/win/UMoncton/Doctorat/dev/dlbd/results/plots/CJCC/")

std_preds = pd.read_feather(src_dir / "std_events_95_15_300_aggregate.feather")

sub_agg = pd.read_feather(src_dir / "sub_events_1_95_aggregate.feather").rename(
    columns={"n_seconds_active": "n_events"}
)
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
    "nesting": {"xlab": "Day", "ylab": "Number of nest initiation/hatch",},
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
    labels={},
):
    df = data.loc[data.site == site]

    lbls = deepcopy(DEFAULT_LABELS)
    lbls.update(labels)
    df_nest = pd.read_excel(nest_data_path)
    df_nest["julian"] = df_nest.date.dt.dayofyear
    df_nest = df_nest.loc[
        (df_nest["plot"] == "brw0"),
    ]

    stages = get_nesting_stages(df_nest)

    xmin = min(stages["incubation"]["start"], df.julian.min())
    xmax = min(stages["hatch"]["end"] + 2, df.julian.max() + 2)

    yrange = df.value.max() - df.value.min()
    bufmin = 0.1 * yrange
    bufmax = 0.15 * yrange
    ymin = df.value.min() - bufmin
    ymax = df.value.max() + bufmax

    se_plot = (
        ggplot(data=df, mapping=aes(x="julian", y="value", colour="site"))
        + xlab(lbls["songs"]["xlab"])
        + ylab(lbls["songs"]["ylab"])
        + geom_point()
        + theme_classic()
        + theme(legend_position="none")
        + geom_smooth(
            method="mavg",
            se=False,
            method_args={"window": 3, "center": True, "min_periods": 1},
        )
        + annotate(
            "rect",
            xmin=[stages["incubation"]["start"], stages["hatch"]["start"]],
            xmax=[stages["incubation"]["end"], stages["hatch"]["end"]],
            ymin=-math.inf,
            ymax=math.inf,
            alpha=0.1,
            fill=["red", "blue"],
        )
        + annotate(
            "text",
            x=[stages["incubation"]["lbl_pos"], stages["hatch"]["lbl_pos"]],
            y=df.value.max() + 0.1 * yrange,
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
        se_plot.save(dest_dir / (se_name + ext), height=8, width=8, dpi=150)
        inc_name = create_file_name("incubation", site, prefix, suffix)
        inc_plot.save(dest_dir / (inc_name + ext), height=8, width=8, dpi=150)
    return (se_plot, inc_plot)


#%%

save_info = {
    "path": dest_dir,
    "height": 10,
    "width": 16,
    "dpi": 150,
    "filename": "cjcc2021_songevents.png",
}

site_data = pd.read_excel(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/sites_deployment_2018.xlsx"
)

plt = EventsPlot(
    events_df=std_preds,
    deployment_data=site_data,
    opts={  # "julian_bounds": (164, 220),
        # "subset": subset,
        # "save": save_info,
        "facet_scales": "fixed",
        "facet_nrow": 2,
        "smoothed": True,
    },
)

plt_sub = EventsPlot(
    events_df=sub_agg,
    deployment_data=site_data,
    opts={  # "julian_bounds": (164, 220),
        # "subset": subset,
        # "save": save,
        "facet_scales": "fixed",
        "facet_nrow": 2,
        "smoothed": True,
    },
)


plt.create_plot_data()
plt_sub.create_plot_data()


#%%

subset = {"include": {"plot": ["BARW_0"]}}

plt_old = EventsPlot(
    events="/mnt/win/UMoncton/Doctorat/dev/ecosongs/src/results/data/song_events_mac2.feather",
    recordings="/mnt/win/UMoncton/Doctorat/dev/ecosongs/src/results/data/recordings_mac.feather",
    deployment_data=site_data,
    opts={  # "julian_bounds": (164, 220),
        "subset": subset,
        # "save": save,
        "facet_scales": "fixed",
        "facet_nrow": 2,
        "smoothed": True,
    },
)

plt_old.create_plot_data()

plt_old.plot()

#%%

# (barw_se, barw_inc) = incubation_plot(
#     plt.plot_data,
#     "Barrow",
#     "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Nest Monitoring/2018/BARW/BARW_nest_2018.xlsx",
#     # save=False,
#     prefix="cjcc2021",
#     dest_dir=dest_dir,
# )

(barw_se, barw_inc) = incubation_plot(
    plt_sub.plot_data,
    "Barrow",
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Nest Monitoring/2018/BARW/BARW_nest_2018.xlsx",
    # save=False,
    prefix="cjcc2021_subsampling_fr",
    dest_dir=dest_dir,
    labels={
        "songs": {
            "xlab": "Jour",
            "ylab": "Nombre moyen de secondes actives par enregistrement (secondes)",
            "periods": ["Initiation de l'incubation", "Éclosion"],
        },
        "nesting": {
            "xlab": "Jour",
            "ylab": "Nombre d'initiations d'incubation / Éclosions",
        },
    },
)

# (barwold_se, barwold_inc) = incubation_plot(
#     plt_old.plot_data,
#     "Barrow",
#     "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Nest Monitoring/2018/BARW/BARW_nest_2018.xlsx",
#     # save=False,
#     prefix="asm2019",
#     dest_dir=dest_dir,
# )


#%%
# (barwold_se, barwold_inc) = incubation_plot(
#     plt_old.plot_data,
#     "Barrow",
#     "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Nest Monitoring/2018/BARW/BARW_nest_2018.xlsx",
#     # save=False,
#     prefix="asm2019_test",
#     dest_dir=dest_dir,
# )

#%%


# def join_tuple(tuple, sep):
#     tuple = list(filter(None, tuple))
#     if len(tuple) > 1:
#         return sep.join(tuple)
#     return tuple[0]


# def check_dates(df, site_data):
#     plot = site_data.loc[site_data["plot"] == df.name]
#     res = df.copy()
#     # res.reset_index()
#     if not plot.empty:
#         start = plot.depl_start.iloc[0]
#         end = plot.depl_end.iloc[0]
#         if not pd.isnull(start):
#             res = res.loc[(df["date"] > start) & (df["date"] < end)]
#         res["lat"] = plot.lat.iloc[0]
#         res["lon"] = plot.lon.iloc[0]
#     # res.ACI = (res.ACI - res.ACI.mean()) / res.ACI.std()
#     return res


# sites = [
#     "Igloolik",
#     "East Bay",
#     "Barrow",
#     "Burntpoint Creek",
#     "Canning River",
#     "Svalbard",
# ]
# plot_excl = ["BARW_8"]

# site_data = pd.read_excel(
#     "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/sites_deployment_2018.xlsx"
# )
# aci = feather.read_dataframe("../plot/data/ACI.feather")
# aci = aci.loc[aci.site.isin(sites)]
# aci = aci.loc[~aci["plot"].isin(plot_excl)]
# aci["julian"] = aci["date"].dt.dayofyear
# aci["hour"] = aci["date"].dt.hour
# aci = aci.sort_values(["site", "plot", "julian", "date"])
# aci = aci.loc[(aci["julian"] > 155) & (aci["julian"] < 220)]
# aci = aci.loc[(aci["ACI"] < 50000)]
# aci = aci.loc[aci["denoised"] == False]

# aci = aci.reset_index()
# res = aci.groupby(["plot"], as_index=False).apply(check_dates, site_data)
# res.index = res.index.droplevel()
# res = res.groupby(["site", "julian"], as_index=False).agg({"ACI": ["mean", "std"]})
# res.columns = pd.Index(join_tuple(i, "_") for i in res.columns)
# aci_test = res
# aci_test["value"] = res.ACI_mean


# (iglo_aci, iglo_inc) = incubation_plot(
#     aci_test,
#     "Igloolik",
#     "/home/vin/Doctorat/data/datasheet/2018/IGLO/IGLO_nest_2018.xlsx",
#     # save=False,
#     prefix="asm2019_ACI2",
# )
# (barw_aci, barw_inc) = incubation_plot(
#     aci_test,
#     "Barrow",
#     "/home/vin/Doctorat/data/datasheet/2018/BARW/BARW_nest_2018.xlsx",
#     # save=False,
#     prefix="asm2019_ACI2",
# )

# plt.plot()
# print(plt.plot_data)
#
# # plt2 = plt.plot()
# # plt2 += theme(legend_position="none",
# #               text=element_text(size=12),
# #               axis_title=element_text(size=14, weight="bold"),
# #               strip_text=element_text(weight="bold"),
# #               plot_title=element_text(linespacing=1.5, size=14, weight="bold", va="center", ha="center", margin={'t': 100, 'b': 15}))
# # plt2 += ggtitle('Mean number and moving average (window =4) of detected bird songs by day and site')
# # plt2.save(**save)
#
# # plt.plots_by_site(filename="plot/figs/All_sites_by_plot_wac.pdf")
