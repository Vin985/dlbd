import librosa
import pandas as pd
from mouffet.data.data_handler import DataHandler
from mouffet.data.data_structure import DataStructure
from mouffet.data.split import split_folder
from mouffet.utils.common import join_tuple
from scipy.ndimage.interpolation import zoom

from ..options.audio_database_options import AudioDatabaseOptions
from . import spectrogram, tag_manager
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

    SPLIT_FUNCS = {"arctic": split_folder, "citynet": citynet_split}

    def get_spectrogram_subfolder_path(self, database, opts):
        return self.get_spec_subfolder(database.spectrogram, opts)

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
            value = str(spec_opts.get(opt, spectrogram.DEFAULT_OPTS[opt]))
            if opt == "type":
                if value == "mel":
                    value += str(spec_opts.get("n_mels", spectrogram.DEFAULT_OPTS[opt]))

            tmp.append(prefix + value)

        spec_folder = "_".join(tmp)
        return spec_folder

    def merge_datasets(self, datasets):
        merged = super().merge_datasets(datasets)
        merged["tags_df"] = pd.concat(merged["tags_df"])
        return merged

    def finalize_dataset(self):
        self.tmp_db_data["tags_df"] = pd.concat(self.tmp_db_data["tags_df"])

    def load_data_options(self, database):
        # db_opts = database.get("options", {})
        opts = {}
        opts["tags"] = database.tags
        opts["spectrogram"] = database.spectrogram
        opts["classes"] = self.load_classes(database)
        return opts

    @staticmethod
    def load_raw_data(file_path, spec_opts, *args, **kwargs):
        sr = spec_opts.get("sample_rate", "original")
        if sr and sr == "original":
            sr = None
        # * NOTE: sample_rate can be different from sr if sr is None
        wav, sample_rate = librosa.load(str(file_path), sr=sr)
        # * NOTE: sp_opts can contain options not defined in spec_opts
        spec, sp_opts = spectrogram.generate_spectrogram(wav, sample_rate, spec_opts)
        audio_info = {
            "file_path": file_path,
            "sample_rate": sample_rate,
            "length": len(wav),
            "spec_opts": sp_opts,
        }
        return spec, audio_info

    @staticmethod
    def load_tags(tags_dir, opts, audio_info, spec_len, *args, **kwargs):
        tag_opts = opts["tags"]
        tag_df = tag_manager.get_tag_df(audio_info, tags_dir, tag_opts)

        # dur = len(wav) / sample_rate
        # print("pps:", spec.shape[1] / dur)

        tmp_tags = tag_manager.filter_classes(tag_df, opts["classes"])
        tag_presence = tag_manager.get_tag_presence(tmp_tags, audio_info, tag_opts)
        factor = float(spec_len) / tag_presence.shape[0]
        zoomed_presence = zoom(tag_presence, factor).astype(int)
        return tmp_tags, zoomed_presence

    def load_file_data(self, file_path, tags_dir, opts):

        spec, audio_info = self.load_raw_data(file_path, opts["spectrogram"])
        tags_df, tags_linear = self.load_tags(tags_dir, opts, audio_info, spec.shape[1])

        self.tmp_db_data["spectrograms"].append(spec)
        self.tmp_db_data["infos"].append(audio_info)
        self.tmp_db_data["tags_df"].append(tags_df)
        self.tmp_db_data["tags_linear_presence"].append(tags_linear)

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
