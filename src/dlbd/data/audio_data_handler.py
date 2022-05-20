import pandas as pd
from mouffet import common_utils
from mouffet.data import DataHandler, DataStructure
from mouffet.data.split import split_folder

from ..options import AudioDatabaseOptions
from . import audio_utils, tag_utils
from .loaders import AudioDataLoader, BADChallengeDataLoader
from .split import citynet_split


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

    def summarize_dataset(self, dataset):
        df, summary = self.summarize_tags(dataset["tags_df"])

        classes_list = list(filter(None, set(df.all_tags.str.split(",").sum())))

        tmp_res = {
            "n_files": len(dataset["spectrograms"]),
            "n_classes": len(df.tag.unique()),
            "classes_summary": summary,
            "classes_list": classes_list,
            "raw_df": df,
        }
        return tmp_res

    def get_db_tags_summary(self, database, db_types=None, filter_classes=False):
        res = {}
        if isinstance(database, str):
            database = self.databases[database]
        db_types = db_types or database.db_types

        paths = self.get_database_paths(database)
        file_lists = self.check_file_lists(database, paths, db_types)

        print("Generating tags summary for database: {}".format(database["name"]))

        res = {}
        tmp_all = []
        # * Only load data if the give db_type is in the database definition
        for db_type in db_types:
            if not db_type in database.db_types:
                continue

            tmp_tags = []
            tags_dir = paths["tags"][db_type]
            for file_path in file_lists[db_type]:
                tmp_df = tag_utils.get_tag_df(file_path, tags_dir, database.tags)
                if filter_classes:
                    tmp_df = tag_utils.filter_classes(tmp_df, database)

                tmp_tags.append(tmp_df)

            tmp_df = pd.concat(tmp_tags)
            tmp_df, tmp_summary = self.summarize_tags(tmp_df)
            res[db_type] = {"raw": tmp_df, "summary": tmp_summary}
            tmp_all.append(tmp_df)

        all_df = pd.concat(tmp_all)
        all_df, all_summary = self.summarize_tags(all_df)
        res["all"] = {"raw": all_df, "summary": all_summary}

        return res

    def get_databases_summary(self, databases, *args, **kwargs):
        res = {}
        databases = databases or self.databases.values()
        # * Iterate over databases
        for database in databases:
            res[database["name"]] = self.get_db_tags_summary(database, *args, **kwargs)
        return res
