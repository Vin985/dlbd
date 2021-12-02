import pandas as pd
from mouffet.data.data_handler import DataHandler
from mouffet.data.data_structure import DataStructure
from mouffet.data.split import split_folder
from mouffet.utils.common import join_tuple

from ..options.audio_database_options import AudioDatabaseOptions
from . import audio_utils
from .split import citynet_split

from .loaders import AudioDataLoader, BADChallengeDataLoader


class AudioDataStructure(DataStructure):
    STRUCTURE = {
        "spectrograms": {"type": "data", "data_type": []},
        "tags_df": {"type": "tags"},
        "tags_linear_presence": {"type": "tags"},
        "infos": {"type": "data"},
    }


class AudioDataHandler(DataHandler):

    OPTIONS_CLASS = AudioDatabaseOptions

    DATA_STRUCTURE = AudioDataStructure()

    DATA_LOADERS = {"default": AudioDataLoader, "bad_challenge": BADChallengeDataLoader}

    SPLIT_FUNCS = {"arctic": split_folder, "citynet": citynet_split}

    def __init__(self, opts):
        super().__init__(opts)

    def get_spectrogram_subfolder_path(self, database, folder_opts=None):
        if folder_opts is None:
            folder_opts = self.get_subfolder_options(database, "spectrogram")
        return self.get_spec_subfolder(database.spectrogram, folder_opts)

    @staticmethod
    def get_subfolder_option_value(opt, opts, default, prefixes):
        return prefixes.get(opt, "") + str(opts.get(opt, default[opt]))

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

    def merge_datasets(self, datasets):
        merged = super().merge_datasets(datasets)
        merged["tags_df"] = pd.concat(merged["tags_df"])
        return merged

    def get_summary(self, dataset):
        df = dataset["tags_df"]
        df["duration"] = df["tag_end"] - df["tag_start"]

        # TODO: FINISH IT!
        summary = df.groupby("tag").agg(
            {"duration": ["sum", "mean", "std", "min", "max"], "tag": "size"}
        )
        summary.columns = pd.Index(join_tuple(i, "_") for i in summary.columns)
        summary = summary.reset_index()
        df.all_tags = df.tag + "," + df.related
        classes_list = list(filter(None, set(df.all_tags.str.split(",").sum())))

        tmp_res = {
            "n_files": len(dataset["spectrograms"]),
            "n_classes": len(df.tag.unique()),
            "classes_summary": summary,
            "classes_list": classes_list,
        }
        return tmp_res
