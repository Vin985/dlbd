import os

import numpy as np
import pandas as pd
from mouffet import common_utils
from scipy.ndimage.interpolation import zoom

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
            "Filename": "file_name",
        },
        "columns_type": {"overlap": "str"},
        "suffix": {"-sceneRect.csv"},
    },
    "nisp4b": {"columns_name": {}, "columns_type": {}},
}


def load_tags(tags_dir, opts, audio_info, spec_len):
    tag_opts = opts["tags"]
    tag_df = get_tag_df(audio_info["file_path"], tags_dir, tag_opts)
    tmp_tags = filter_classes(tag_df, opts)
    tag_presence = get_tag_presence(tmp_tags, audio_info, tag_opts)
    factor = float(spec_len) / tag_presence.shape[0]
    zoomed_presence = zoom(tag_presence, factor).astype(int)
    return tmp_tags, zoomed_presence


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


def filter_classes(tag_df, opts):
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
    classes = opts["classes"]
    if tag_df.empty or not opts["tags"].get("filter_classes", True):
        return tag_df
    tag_df = tag_df.dropna(subset=["tag"])
    if "related" not in tag_df.columns:
        tag_df["related"] = ""
    tag_df.loc[tag_df.related.isnull(), "related"] = ""
    df2 = tag_df[["tag", "related"]]
    res = [False] * tag_df.shape[0]
    match = tag_df.tag.values
    i = 0
    for tag, related in df2.itertuples(name=None, index=False):
        found = False
        tag = tag.lower()
        if tag in classes:
            found = True
        else:
            if not found and opts["tags"].get("print_tag_not_found", False):
                common_utils.print_warning(
                    "Tag {} not found in accepted classes, searching in related tags".format(
                        tag
                    )
                )
            related_tags = related.lower().split(",")
            for c in classes:
                if c in related_tags:
                    found = True
                    match[i] = c
                    break
            # * If we arrived here, class was not found
        if not found and opts["tags"].get("print_missing_classes", False):
            common_utils.print_warning(
                "class {} not present in accepted classes, skipping".format(tag)
            )
        res[i] = found
        i += 1
    tag_df.loc[:, "tag"] = match
    return tag_df[res]


def get_audiotagger_tag_df(audio_file_path, labels_dir, tag_opts):
    defaults = DEFAULT_OPTIONS["audiotagger"]
    columns = tag_opts["columns"] or defaults["columns_name"]
    columns_type = tag_opts["columns_type"] or defaults["columns_type"]
    suffix = tag_opts.get("suffix", defaults["suffix"])

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
        # tag_df = pd_annots.loc[~pd_annots.Filename.isna()]..copy()
        tag_df = rename_columns(pd_annots, columns)
        if "file_name" not in tag_df:
            tag_df["file_name"] = str(audio_file_path.stem) + ".wav"
        tag_df = tag_df.loc[~tag_df.file_name.isna()]

        return tag_df
    else:
        print("Warning - no annotations found for %s" % str(audio_file_path))
        return pd.DataFrame()


def get_nips4b_tag_df(audio_file_path, labels_dir, tag_opts):
    file_id = audio_file_path.stem[-3:]
    tag_path = labels_dir / ("annotation_train" + file_id + ".csv")
    if tag_path.exists():
        print("Loading tag file: " + str(tag_path))
        tag_df = pd.read_csv(tag_path, names=["tag_start", "tag_duration", "tag"])
        tag_df["tag_end"] = tag_df["tag_start"] + tag_df["tag_duration"]

        tag_df.loc[
            tag_df["tag"] == "Unknown", "tag"  # pylint: disable=unsubscriptable-object
        ] = "UNKN"
        return tag_df
    else:
        print("Warning - no annotations found for %s" % str(audio_file_path))
        return pd.DataFrame()


def get_bad_challenge_tag_df(tags_dir):
    tag_path = tags_dir / "tags.csv"
    if tag_path.exists():
        print("Loading tag file: " + str(tag_path))
        tag_df = pd.read_csv(tag_path)
        return tag_df
    else:
        raise ValueError("Error - no annotations file found %s" % str(tag_path))


def get_tag_df(audio_file_path, labels_dir, tag_opts):
    tag_type = tag_opts.get("type", "audiotagger")

    tag_func_name = "get_" + tag_type + "_tag_df"

    possibles = globals().copy()
    possibles.update(locals())
    func = possibles.get(tag_func_name)

    tag_df = func(audio_file_path, labels_dir, tag_opts)
    if not tag_df.empty:
        tag_df.loc[:, "recording_path"] = str(audio_file_path)
    return tag_df


def get_tag_presence(tag_df, audio_info, tag_opts):
    tag_presence = np.zeros(audio_info["length"])
    for _, annot in tag_df.iterrows():
        # * fill in the label vector
        start_point = int(float(annot["tag_start"]) * audio_info["sample_rate"])
        end_point = int(float(annot["tag_end"]) * audio_info["sample_rate"])

        tag_presence[start_point:end_point] = 1
    return tag_presence


def prepare_tags(tags):
    tags = tags.astype({"recording_path": "category"})
    tags["tag_duration"] = tags["tag_end"] - tags["tag_start"]
    tags.reset_index(inplace=True)
    tags.rename(
        columns={"index": "tag_index", "recording_path": "recording_id"},
        inplace=True,
    )
    tags.reset_index(inplace=True)
    tags.rename(columns={"index": "id"}, inplace=True)
    return tags


def flatten_tags(tags_df):
    """Flattens all tags, meaning reduce all overlapping tags to a single tag which starts at the
    beginning of the first tag and stops at the end of the last tag

    Args:
        tags_df (pandas.DataFrame): A dataframe with all the tags. Works with tags in the
        AudioTagger format

    Returns:
        pandas.DataFrame: A DataFrame with all tags flattened
    """
    res = []
    previous = {}
    tags_df.sort_values(by=["tag_start"])
    for _, tag in tags_df.iterrows():
        if not previous:
            previous = {
                "tag_start": tag.tag_start,
                "tag_end": tag.tag_end,
                "tag_duration": tag.tag_duration,
                "n_tags": 1,
            }
            res.append(previous)
        else:
            # * Next tag overlaps with the previous one
            if previous["tag_start"] <= tag.tag_start < previous["tag_end"]:
                # * Tag ends later than the previous one: use the end of this one instead
                if tag.tag_end > previous["tag_end"]:
                    previous["tag_end"] = tag.tag_end
                    previous["n_tags"] += 1
                    previous["tag_duration"] = (
                        previous["tag_end"] - previous["tag_start"]
                    )
            else:
                # * Tag starts after the next one, create new tag
                previous = {
                    "tag_start": tag.tag_start,
                    "tag_end": tag.tag_end,
                    "tag_duration": tag.tag_duration,
                    "n_tags": 1,
                }
                res.append(previous)
    res = pd.DataFrame(res)
    return res


def plot_summary(tags, opts=None):
    from plotnine import (
        aes,
        element_text,
        geom_bar,
        geom_text,
        ggplot,
        theme,
        theme_classic,
        xlab,
        ylab,
    )
    from plotnine.positions.position_dodge import position_dodge

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
