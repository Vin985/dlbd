import math

import feather
import pandas as pd
from plotnine import (
    aes,
    facet_wrap,
    geom_line,
    geom_point,
    geom_smooth,
    ggplot,
    save_as_pdf_pages,
    scale_colour_manual,
    scale_x_continuous,
    theme,
    xlab,
    ylab,
)
from plotnine.facets.facet_wrap import n2mfrow

from .plot import Plot
from .utils import exclude_rows, get_rows, join_tuple, label_x


class EventsPlot(Plot):
    def __init__(
        self,
        events=None,
        recordings=None,
        plot_data=None,
        opts=None,
        deployment_data=None,
        events_df=None,
    ):
        super().__init__(plot_data, opts)
        self.events_raw = events
        self.recordings_raw = recordings
        self.deployment_data_raw = deployment_data
        self.events = events_df
        self.plt = None

    def create_plot_data(self):
        if self.events is None and not self.events_raw and not self.recordings_raw:
            raise AttributeError(
                (
                    'Either "plot_data" with all relevant information'
                    " for plotting, paths to files containing"
                    " events and recordings data"
                    " or dataframes with this information should be provided"
                )
            )
        if self.events is None:
            self.get_events()
        plot_data = self.events.copy()
        plot_data = self.subset_data(plot_data)
        plot_data = plot_data.groupby(["plot"], as_index=False).apply(self.check_dates)
        plot_data = self.check_data_columns(plot_data)
        plot_data = self.aggregate(plot_data, "n_events")
        self.plot_data = plot_data

    def get_data(self, name):
        data = getattr(self, name + "_raw")
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, str):
                # events is a path, load data file
                path = data
                data = feather.read_dataframe(path)
                setattr(self, name + "_path", path)
                setattr(self, name + "_raw", data)
            else:
                print("Unsupported type provided for " + name)
                return None
        return data

    def get_events(self):
        # verify that events are correctly loaded
        self.get_data("events")
        self.get_data("recordings")
        self.aggregate_events()
        self.add_recording_info()

    def subset_data(self, data, subset_opts=None):
        opts = subset_opts or self.opts.get("subset", None)
        if not opts:
            return data

        if opts.get("include"):
            data = get_rows(data, opts.get("include"), include=True)
        elif opts.get("exclude"):
            data = get_rows(data, opts.get("exclude"), include=False)

        return data

    def count_events(self):
        events = self.events_raw
        if not "n_events" in events.columns:
            events = self.events_raw.groupby(["recording_id"], as_index=False).agg(
                {"event_id": "count"}
            )
            events.rename(columns={"event_id": "n_events"}, inplace=True)
        self.events = events

    def aggregate_events(self, by="count", on=["recording_id"]):

        # agg_by = self.opts.get("agg_events_by", by)
        # agg_on = self.opts.get("agg_events_on", on)
        #
        # res = self.events_raw.groupby(agg_on, as_index=False).agg({"event_id": agg_by})
        # res.rename(columns={'event_id': 'n_events'}, inplace=True)
        # TODO change aggregation options
        if by == "count":
            self.count_events()
        # return self.events_raw

    def add_recording_info(self):
        # TODO: externalize columns selection
        recs = self.recordings_raw[["id", "date", "name", "site", "plot"]]
        self.events = self.events.merge(recs, left_on="recording_id", right_on="id")

    def check_data_columns(self, data):
        # if "duration" not in data:
        #     data["duration"] = data["end"] - data["start"]
        if "julian" not in data:
            data["julian"] = data["date"].dt.dayofyear
        if "hour" not in data:
            data["hour"] = data["date"].dt.hour
        data = data.sort_values(["site", "plot", "julian", "date"])
        if "julian_bounds" in self.opts:
            min_j, max_j = self.opts["julian_bounds"]
            data = data.loc[(data["julian"] > min_j) & (data["julian"] < max_j)]
        return data

    def check_dates(self, df):
        # TODO: adapt code to current data handling
        if self.deployment_data_raw is not None:
            site_data = self.get_data("deployment_data")
            plot = site_data.loc[site_data["plot"] == df.name]
            res = df
            if not plot.empty:
                start = plot.depl_start.iloc[0]
                end = plot.depl_end.iloc[0]
                if not pd.isnull(start):
                    res = df.loc[(df["date"] > start) & (df["date"] < end)]
                res["lat"] = plot.lat.iloc[0]
                res["lon"] = plot.lon.iloc[0]
            return res
        return df

    def aggregate(
        self,
        data,
        column,
        group_by=["site", "julian", "type"],
        agg_by={"value": "mean"},
    ):
        group_by = self.opts.get("plt_group_by", group_by)
        agg_by = self.opts.get("plt_agg_by", agg_by)
        data["type"] = column
        data.rename(columns={column: "value"}, inplace=True)
        res = data.groupby(group_by, as_index=False).agg(agg_by)
        # res.columns = pd.Index(join_tuple(i, "_") for i in res.columns)
        return res

    def get_facet_rows(self, data, facet_by):
        nrow = self.opts.get("facet_nrow", None)
        ncol = self.opts.get("facet_ncol", None)
        if not nrow and not ncol:
            nplots = len(data[facet_by].unique())
            nrow, ncol = n2mfrow(nplots)
        return (nrow, ncol)

    def plot_plots(self, site, plot_options, as_pdf):
        subset = self.plot_data.loc[self.plot_data["site"] == site]
        plots = subset["plot"].unique()
        if len(plots) > 0:
            if as_pdf:
                plot_options["save"] = None
            else:
                plot_options["save"]["filename"] = "Plots_" + site
            return self.__plot(subset, **plot_options)
        return None

    def plots_by_site(self, as_pdf=True, filename="plot/figs/All_sites_by_plot.pdf"):
        self.opts["plt_group_by"] = ["site", "plot", "julian", "type"]
        if self.plot_data is None:
            self.create_plot_data()
        sites = self.plot_data["site"].unique()
        # Update plot options
        plot_options = self.get_plot_options()
        plot_options["colour"] = "plot"
        plot_options["facet_by"] = "plot"
        plots = [self.plot_plots(site, plot_options, as_pdf) for site in sites]
        plots = [plot for plot in plots if plot is not None]
        if as_pdf:
            save_as_pdf_pages(plots, filename)

    def get_plot_options(self):
        opts = {}
        opts["x"] = self.opts.get("x", "julian")
        opts["y"] = self.opts.get("y", "value")
        opts["colour"] = self.opts.get("colour", "site")
        opts["lbl_x"] = self.opts.get("xlab", "Day")
        opts["lbl_y"] = self.opts.get("ylab", "Mean number of detected songs")

        opts["facet"] = self.opts.get("facet", True)
        opts["facet_by"] = self.opts.get("facet_by", "site")
        opts["facet_scales"] = self.opts.get("facet_scales", "free_y")

        opts["error_bars"] = self.opts.get("error_bars", False)

        opts["smoothed"] = self.opts.get("smoothed", True)
        opts["points"] = self.opts.get("points", True)
        opts["save"] = self.opts.get("save", None)
        return opts

    def plot(self):
        if self.plot_data is None:
            self.create_plot_data()

        plot_opts = self.get_plot_options()
        # x = self.opts.get("x", "julian")
        # y = self.opts.get("y", "value")
        # colour = self.opts.get("colour", "site")
        # lbl_x = self.opts.get("xlab", "Day")
        # lbl_y = self.opts.get("ylab", "Mean number of detected songs")
        #
        # facet = self.opts.get("facet", True)
        # facet_by = self.opts.get("facet_by", "site")
        # facet_scales = self.opts.get("facet_scales", "free_y")
        #
        # smoothed = self.opts.get("smoothed", True)
        # points = self.opts.get("points", True)
        # save = self.opts.get("save", None)

        self.plt = self.__plot(self.plot_data, **plot_opts)
        return self.plt
        # cbbPalette = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]

    def __plot(
        self,
        plot_data,
        x,
        y,
        colour,
        lbl_x,
        lbl_y,
        facet,
        facet_scales,
        facet_by,
        smoothed,
        points,
        error_bars,
        save,
    ):
        cbbPalette = [
            "#000000",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
        ]
        plt = ggplot(data=plot_data, mapping=aes(x=x, y=y, colour=colour))
        plt += xlab(lbl_x)
        plt += ylab(lbl_y)
        # + facet_grid("site~", scales="free")
        # + geom_line()
        if facet:
            # TODO: use facet as save
            nrow, ncol = self.get_facet_rows(plot_data, facet_by)
            plt += facet_wrap(facet_by, nrow=nrow, ncol=ncol, scales=facet_scales)
        if points:
            plt += geom_point()
        if error_bars:
            # TODO use generic way to compute them
            pass
            # self.plt += geom_errorbar(aes(ymin="ACI_mean - ACI_std", ymax="ACI_mean + ACI_std"))
        # TODO: use smooth as save
        if smoothed:
            plt += geom_smooth(
                method="mavg",
                se=False,
                method_args={"window": 4, "center": True, "min_periods": 1},
            )
        else:
            plt += geom_line()
        plt += scale_colour_manual(values=cbbPalette, guide=False)
        plt += scale_x_continuous(labels=label_x)

        plt += theme(figure_size=(15, 18), dpi=150)

        if save:
            plt.save(**save)
        return plt
