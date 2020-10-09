import os

import numpy as np
import pandas as pd

DEFAULT_TAGS_COLUMNS = {
    "Label": "tag",
    "Related": "related",
    "LabelStartTime_Seconds": "tag_start",
    "LabelEndTime_Seconds": "tag_end",
    "overlap": "overlap",
    "background": "background",
    "noise": "noise",
}
DEFAULT_TAGS_COLUMNS_TYPE = {"overlap": "str"}


def rename_columns(df, columns):
    if columns:
        if isinstance(columns, dict):
            # df["tags"] = df["Label"] + \
            #     "," + df["Related"]
            df = df[columns.keys()]
            df = df.rename(columns=columns)
        else:
            raise ValueError(
                "columns must be a dict with old labels as keys and new labels as values"
            )
    return df


def load_tags(audio_info, labels_dir, tag_opts):

    columns = tag_opts["columns"] or DEFAULT_TAGS_COLUMNS
    columns_type = tag_opts["columns_type"] or DEFAULT_TAGS_COLUMNS_TYPE
    suffix = tag_opts["suffix"]
    audio_file_path = audio_info["file_path"]

    if tag_opts["tags_with_audio"]:
        csv_file_path = audio_file_path.parent / (audio_file_path.stem + suffix)
    else:
        csv_file_path = labels_dir / (audio_file_path.stem + suffix)
    tags = np.zeros(audio_info["length"])
    print("Loading tags for file: " + str(audio_file_path))
    if os.path.exists(csv_file_path):
        pd_annots = pd.read_csv(
            csv_file_path, skip_blank_lines=True, dtype=columns_type
        )
        # loop over each annotation...
        tmp = pd_annots.loc[~pd_annots.Filename.isna()]
        if tag_opts["as_df"]:
            tmp = rename_columns(tmp, columns)
            # tmp.loc[:, "recording_id"] = audio_info["recording_id"]
            if not tmp.empty:
                tmp.loc[:, "recording_path"] = str(audio_info["file_path"])
            return tmp

        # create label vector...
        for _, annot in tmp.iterrows():
            # fill in the label vector
            start_point = int(
                float(annot["LabelStartTime_Seconds"]) * audio_info["sample_rate"]
            )
            end_point = int(
                float(annot["LabelEndTime_Seconds"]) * audio_info["sample_rate"]
            )

            label = annot["Label"].lower()
            if label in tag_opts["classes"]:
                tags[start_point:end_point] = 1
    else:
        print("Warning - no annotations found for %s" % str(audio_file_path))

    return tags
