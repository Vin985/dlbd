from plotnine import (
    ggplot,
    geom_line,
    ggtitle,
    xlab,
    ylab,
    scale_color_discrete,
    scale_x_datetime,
    theme_classic,
    theme,
    element_text,
    geom_point,
    aes,
    ylim,
    scale_color_manual,
    element_blank,
)
from .utils import format_date_short


def plot_real_distance(data, options, infos, **plot_args):
    plot_args["y"] = "trend"
    if "title" not in plot_args:
        plot_args["title"] = (
            f"Daily mean activity per recording with method {options['method']}.\n"
            + f" Euclidean distance to reference: {data['distance']}"
        )
    return plot_distance(data["df"], options, infos, **plot_args)


def plot_norm_distance(data, options, infos, **plot_args):
    plot_args["y"] = "trend_norm"
    if "title" not in plot_args:
        plot_args["title"] = (
            f"Normalized daily mean activity per recording with method {options['method']}.\n"
            + f" Euclidean distance to reference: {data['distance_norm']}"
        )
    return plot_distance(data["df"], options, infos, **plot_args)


def plot_distance(data, options, infos, **plot_args):
    tmp_plt = (
        ggplot(
            data=data,
            mapping=aes(
                plot_args.get("x", "date"), plot_args.get("y", "trend"), color="type"
            ),
        )
        + geom_line()
        + ggtitle(plot_args.get("title", ""))
        + xlab(plot_args.get("xlab", "Date"))
        + ylab(plot_args.get("ylab", "Daily mean activity per recording (s)"))
        + scale_color_discrete(labels=["Model", "Reference"])
        + scale_x_datetime(labels=format_date_short)
        + theme_classic()
        + theme(axis_text_x=element_text(angle=45))
    )
    if options.get("add_points", False):
        tmp_plt = tmp_plt + geom_point(mapping=aes(y="total_duration"))
    return tmp_plt


def plot_distances(data, options, infos, **plot_args):
    res = []
    if options.get("plot_real_distance", True):
        tmp_plot = plot_real_distance(data, options, infos, **plot_args)
        res.append(tmp_plot)
    if options.get("plot_norm_distance", True):
        tmp_plt_norm = plot_norm_distance(data, options, infos, **plot_args)
        res.append(tmp_plt_norm)
        # tmp_plt_norm = (
        #     ggplot(
        #         data=plt_df,
        #         mapping=aes("date", "trend_norm", color="type"),
        #     )
        #     + geom_line()
        #     + ggtitle(
        #         f"Normalized daily mean activity per recording with method {options['method']}.\n"
        #         + f" Euclidean distance to reference: {data['distance_norm']}"
        #     )
        #     + xlab("Date")
        #     + ylab("Normalized daily mean activity per recording")
        #     + scale_color_discrete(labels=["Model", "Reference"])
        #     + scale_x_datetime(labels=format_date_short)
        #     + theme_classic()
        #     + theme(axis_text_x=element_text(angle=45))
        # )
        # res.append(tmp_plt_norm)
    return res


def plot_separate_distances(data, options, infos):
    plt_df = data["df"]
    res = []
    y_range = max(plt_df.trend_norm) - min(plt_df.trend_norm)
    ylims = [
        min(plt_df.trend_norm) - 0.1 * y_range,
        max(plt_df.trend_norm) + 0.1 * y_range,
    ]
    plt_gt_norm = (
        ggplot(
            data=plt_df.loc[plt_df.type == "ground_truth"],
            mapping=aes("date", "trend_norm", color=["#0571b0"]),
        )
        + geom_line(size=1.5)
        + xlab("Date")
        + ylab("Normalized vocal activity")
        + ggtitle(
            "Normalized trends for model {} on database {} with method {}.\n".format(
                options["scenario_info"]["model"],
                options["scenario_info"]["database"],
                options["method"],
            )
            + " Euclidean distance to reference: {} \n".format(data["distance_norm"])
        )
        + ylim(ylims)
        + scale_color_manual(values=["#0571b0"], labels=["Reference"])
        + scale_x_datetime(labels=format_date_short)
        + theme_classic()
        + theme(
            title=element_text(face="bold"),
            axis_text_x=element_text(angle=45),
            axis_text=element_text(face="bold"),
            legend_title=element_blank(),
            axis_title=element_text(face="bold"),
        )
    )

    plt_norm = (
        ggplot(
            data=plt_df,
            mapping=aes("date", "trend_norm", color="type"),
        )
        + geom_line(size=1.5)
        + ggtitle(
            "Normalized trends for model {} on database {} with method {}.\n".format(
                options["scenario_info"]["model"],
                options["scenario_info"]["database"],
                options["method"],
            )
            + " Euclidean distance to reference: {} \n".format(data["distance_norm"])
        )
        + xlab("Date")
        + ylab(
            "Normalized vocal activity",
        )
        + ylim(ylims)
        + scale_color_manual(
            values=["#0571b0", "#f4a582"], labels=["Reference", "Model"]
        )
        + scale_x_datetime(labels=format_date_short)
        + theme_classic()
        + theme(
            title=element_text(face="bold"),
            axis_text_x=element_text(angle=45),
            axis_text=element_text(face="bold"),
            legend_title=element_blank(),
            axis_title=element_text(face="bold"),
        )
    )
    res.append(plt_gt_norm)
    res.append(plt_norm)

    return res
