import os

import numpy as np
import pandas as pd

from plotnine import (
    aes,
    geom_bar,
    geom_text,
    ggplot,
    scale_x_discrete,
    element_text,
    theme_classic,
    theme,
    xlab,
    ylab,
)
from plotnine.positions.position_dodge import position_dodge

DEFAULT_OPTIONS = {
    "audiotagger": {
        "columns_name": {
            "Label": "tag",
            "Related": "related",
            "LabelStartTime_Seconds": "tag_start",
            "LabelEndTime_Seconds": "tag_end",
            "overlap": "overlap",
            "background": "background",
            "noise": "noise",
        },
        "columns_type": {"overlap": "str"},
        "suffix": {"-sceneRect.csv"},
    },
    "nisp4b": {"columns_name": {}, "columns_type": {}},
}


def rename_columns(df, columns):
    if columns:
        if isinstance(columns, dict):
            # df["tags"] = df["Label"] + \
            #     "," + df["Related"]
            keys = [column for column in df.columns if column in columns.keys()]
            df = df[keys]
            df = df.rename(columns=columns)
        else:
            raise ValueError(
                "columns must be a dict with old labels as keys and new labels as values"
            )
    return df


def filter_classes(tag_df, classes):
    """Filters the provided tag dataframe to select only tags in the classes list.
    If no match was found in the 'tag' column and a 'related' column is present in the dataframe, 
    this column will be also checked. If a match is found in the 'related' column,
    then value of the 'tag' column will be replaced by the match.
    Matches are decided by the order of the classes list and only the first match will be returned.

    Args:
        tag_df (pandas.Dataframe): Dataframe containing the tag information. Note that at least a
        'tag' column with the label value must be present
        classes (list): list of classes we want to filter the dataframe on

    Returns:
        pandas.Dataframe: filtered and possibly altered dataframe with 'tag' values only in list
    """
    if tag_df.empty:
        return tag_df
    if "related" not in tag_df.columns:
        tag_df["related"] = ""
    tag_df.loc[tag_df.related.isnull(), "related"] = ""
    df2 = tag_df[["tag", "related"]]
    res = [False] * tag_df.shape[0]
    match = tag_df.tag.values
    i = 0
    for tag, related in df2.itertuples(name=None, index=False):
        tag = tag.lower()
        if tag in classes:
            res[i] = True
        else:
            related_tags = related.lower().split(",")
            for c in classes:
                if c in related_tags:
                    res[i] = True
                    match[i] = c
                    break
        i += 1
    tag_df.loc[:, "tag"] = match
    return tag_df[res]


def get_audiotagger_tag_df(audio_info, labels_dir, tag_opts):
    defaults = DEFAULT_OPTIONS["audiotagger"]
    columns = tag_opts["columns"] or defaults["columns_name"]
    columns_type = tag_opts["columns_type"] or defaults["columns_type"]
    suffix = tag_opts.get("suffix", defaults["suffix"])

    audio_file_path = audio_info["file_path"]

    if tag_opts.get("with_data", False):
        csv_file_path = audio_file_path.parent / (audio_file_path.stem + suffix)
    else:
        csv_file_path = labels_dir / (audio_file_path.stem + suffix)
    print("Trying to load tag file: " + str(csv_file_path))
    if os.path.exists(csv_file_path):
        pd_annots = pd.read_csv(
            csv_file_path, skip_blank_lines=True, dtype=columns_type
        )
        # * loop over each annotation...
        tag_df = pd_annots.loc[~pd_annots.Filename.isna()].copy()
        tag_df = rename_columns(tag_df, columns)
        return tag_df
    else:
        print("Warning - no annotations found for %s" % str(audio_file_path))
        return pd.DataFrame()


def get_nips4b_tag_df(audio_info, labels_dir, tag_opts):
    audio_file_path = audio_info["file_path"]
    file_id = audio_file_path.stem[-3:]
    tag_path = labels_dir / ("annotation_train" + file_id + ".csv")
    if tag_path.exists():
        print("Loading tag file: " + str(tag_path))
        tag_df = pd.read_csv(tag_path, names=["tag_start", "tag_duration", "tag"])
        tag_df["tag_end"] = tag_df["tag_start"] + tag_df["tag_duration"]
        return tag_df
    else:
        print("Warning - no annotations found for %s" % str(audio_file_path))
        return pd.DataFrame()


def get_tag_df(audio_info, labels_dir, tag_opts):
    tag_type = tag_opts.get("type", "audiotagger")

    tag_func_name = "get_" + tag_type + "_tag_df"

    possibles = globals().copy()
    possibles.update(locals())
    func = possibles.get(tag_func_name)

    tag_df = func(audio_info, labels_dir, tag_opts)
    if not tag_df.empty:
        tag_df.loc[:, "recording_path"] = str(audio_info["file_path"])
    return tag_df


def get_tag_presence(tag_df, audio_info, tag_opts):
    tag_presence = np.zeros(audio_info["length"])
    for _, annot in tag_df.iterrows():
        # * fill in the label vector
        start_point = int(float(annot["tag_start"]) * audio_info["sample_rate"])
        end_point = int(float(annot["tag_end"]) * audio_info["sample_rate"])

        tag_presence[start_point:end_point] = 1
    return tag_presence


def summary(tags, opts=None):
    print(tags)
    tags_summary = (
        tags.groupby(["tag", "background"])
        .agg({"tag": "count"})
        .rename(columns={"tag": "n_tags"})
        .reset_index()
        .astype({"background": "category", "tag": "category"})
    )
    print(tags_summary)
    # tags_summary = tags_df.groupby(["species"]).agg(
    #     {"tag_duration": "sum", "species": "count"}
    # )

    # tags_summary.rename(columns={"species": "count"}, inplace=True)

    # tags_summary["tag_duration"] = tags_summary.tag_duration.astype(int)
    # tags_summary["duration"] = tags_summary.tag_duration.astype(str) + "s"
    # tags_summary = tags_summary.reindex(list(SPECIES_LABELS.keys()))
    # # tags_summary["species"] = tags_summary.index
    # tags_summary.reset_index(inplace=True)
    # tags_summary
    # (
    #     ggplot(
    #         data=tags_summary,
    #         mapping=aes(
    #             x="factor(species, ordered=False)",
    #             y="tag_duration",
    #             fill="factor(species, ordered=False)",
    #         ),
    #     )
    #     + geom_bar(stat="identity", show_legend=False)
    #     + xlab("Species")
    #     + ylab("Duration of annotations (s)")
    #     + geom_text(mapping=aes(label="count"), nudge_y=15)
    #     + theme_classic()
    #     + scale_x_discrete(limits=SPECIES_LIST, labels=xlabels)
    # ).save("species_repartition_duration_mini.png", width=10, height=8)

    plt = (
        ggplot(
            data=tags_summary,
            mapping=aes(
                x="tag",  # "factor(species, ordered=False)",
                y="n_tags",
                fill="background",  # "factor(species, ordered=False)",
            ),
        )
        + geom_bar(stat="identity", show_legend=True, position=position_dodge())
        + xlab("Species")
        + ylab("Number of annotations")
        + geom_text(mapping=aes(label="n_tags"), nudge_y=15)
        + theme_classic()
        + theme(axis_text_x=element_text(angle=90, vjust=1, hjust=1, margin={"r": -30}))
        # + scale_x_discrete(limits=SPECIES_LIST, labels=xlabels)
    ).save("tag_species_bg.png", width=10, height=8)
    # print(tags_summary)

    print(plt)

    # xlabels = [lab.replace(" ", "\n") for lab in SPECIES_LABELS.values()]
    # xlabels
    # plt = (
    #     ggplot(
    #         data=tags_df,
    #         mapping=aes(
    #             x="factor(species, ordered=False)",
    #             fill="factor(species, ordered=False)",
    #         ),
    #     )
    #     # , width=0.4,    position=position_dodge(width=0.5))
    #     + xlab("Species")
    #     + ylab("Number of annotations")
    #     + geom_bar(show_legend=False)
    #     + theme(axis_title=element_text(size=18), axis_text=element_text(size=10))
    #     # + theme(legend_title="Species")
    #     # + scale_fill_discrete(guide=False, limits=SPECIES_LIST,
    #     #                       labels=list(SPECIES_LABELS.values()))
    #     + scale_x_discrete(limits=SPECIES_LIST, labels=xlabels)
    # )
    # print(plt)
    # plt.save("species_repartition_all.png", width=10, height=8)
