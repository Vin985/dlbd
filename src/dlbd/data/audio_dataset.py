from mouffet.data import Dataset
import pandas as pd

from . import audio_utils, tag_utils
from mouffet import common_utils
from .loaders import AudioDataLoader, BADChallengeDataLoader


class AudioDataset(Dataset):

    STRUCTURE = {
        "spectrograms": {"type": "data", "data_type": []},
        "spec_opts": {"type": "data"},
        "tags_df": {"type": "tags"},
        "tags_linear_presence": {"type": "tags"},
        "metadata": {"type": "data"},
    }

    LOADERS = {"default": AudioDataLoader, "bad_challenge": BADChallengeDataLoader}

    def get_spec_subfolder(self, spec_opts, folder_opts):

        opts = folder_opts.get(
            "options", ["sample_rate", "type", "n_fft", "win_length", "hop_length"]
        )
        prefixes = folder_opts.get("prefixes", {})
        tmp = []
        for opt in opts:
            prefix = str(prefixes.get(opt, ""))
            value = str(spec_opts.get(opt, audio_utils.DEFAULT_SPEC_OPTS[opt]))
            if opt == "type":
                if value == "mel":
                    value += str(
                        spec_opts.get("n_mels", audio_utils.DEFAULT_SPEC_OPTS[opt])
                    )

            tmp.append(prefix + value)

        spec_folder = "_".join(tmp)
        return spec_folder

    def tags_file_name(self, key, db_type, database):
        filter_classes = database.get("tags", {}).get("filter_classes", True)
        if filter_classes and not database.class_type:
            filter_classes = False
        class_type = database.class_type if filter_classes else "no_filter"
        return db_type + "_" + key + "_" + class_type + "." + self.get_extension(key)

    def metadata_subfolders(self, key):
        return ""

    def tags_df_subfolders(self, key):
        return ""

    def get_spectrogram_subfolder_path(self, folder_opts=None):
        if folder_opts is None:
            folder_opts = self.get_subfolder_options("spectrogram")
        return self.get_spec_subfolder(self.database.spectrogram, folder_opts)

    @staticmethod
    def summarize_tags(df):
        df["duration"] = df["tag_end"] - df["tag_start"]
        df["all_tags"] = df.tag
        if "related" in df.columns:
            df["all_tags"] += "," + df.related
        summary = df.groupby("tag").agg(
            {
                "duration": ["sum", "mean", "std", "min", "max"],
                "tag": "size",
                "all_tags": "first",
            }
        )
        summary.columns = pd.Index(
            common_utils.join_tuple(i, "_") for i in summary.columns
        )
        summary = summary.reset_index()
        return df, summary

    def summarize(self):
        df, summary = self.summarize_tags(self.data["tags_df"])

        classes_list = list(filter(None, set(df.all_tags.str.split(",").sum())))

        durations = [info["duration"] for info in self.data["metadata"]]
        flattened = (
            df.groupby("recording_id").apply(tag_utils.flatten_tags).reset_index()
        )

        tmp_res = {
            "n_files": len(self.data["metadata"]),
            "total_audio_duration": int(sum(durations)),
            "total_time_active": int(flattened.tag_duration.sum()),
            "n_annotations": df.shape[0],
            "n_classes": len(df.tag.unique()),
            "classes_summary": summary,
            "classes_list": classes_list,
            "raw_df": df,
        }
        return tmp_res

    def get_ground_truth(self):
        return self.data["tags_linear_presence"]

    def get_raw_data(self):
        return self.data["spectrograms"]
