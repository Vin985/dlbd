import librosa
import pandas as pd
from dlbd.options.audio_database_options import AudioDatabaseOptions
from mouffet.data.data_handler import DataHandler
from scipy.ndimage.interpolation import zoom

from . import spectrogram, tag_manager


class AudioDataHandler(DataHandler):

    OPTIONS_CLASS = AudioDatabaseOptions

    DATA_STRUCTURE = {
        "spectrograms": [],
        "tags_df": [],
        "tags_linear_presence": [],
        "infos": [],
    }

    def get_spectrogram_subfolder_path(self, database):
        return spectrogram.get_spec_subfolder(database.spectrogram)

    def merge_datasets(self, datasets):
        merged = super().merge_datasets(datasets)
        merged["tags_df"] = pd.concat(merged["tags_df"])
        return merged

    def finalize_dataset(self):
        self.tmp_db_data["tags_df"] = pd.concat(self.tmp_db_data["tags_df"])

    def load_data_options(self, database):
        # db_opts = database.get("options", {})
        opts = {}
        opts["tags"] = database.tags  # self.load_option_group("tags", db_opts)
        opts[
            "spectrogram"
        ] = database.spectrogram  # self.load_option_group("spectrogram", db_opts)
        opts["classes"] = self.load_classes(database)
        return opts

    @staticmethod
    def load_raw_data(file_path, opts, *args, **kwargs):
        spec_opts = opts["spectrogram"]
        sr = spec_opts.get("sample_rate", "original")
        if sr and sr == "original":
            sr = None
        # * NOTE: sample_rate can be different from sr if sr is None
        wav, sample_rate = librosa.load(str(file_path), sr=sr)
        # * NOTE: sp_opts can contain options not defined in spec_opts
        spec, sp_opts = spectrogram.generate_spectrogram(wav, sample_rate, spec_opts)
        return spec, sp_opts, sample_rate, len(wav)

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
        spec, spec_opts, sr, nframes = self.load_raw_data(file_path, opts)

        audio_info = {
            "file_path": file_path,
            "sample_rate": sr,
            "length": nframes,
            "spec_opts": spec_opts,
        }
        tags_df, tags_linear = self.load_tags(tags_dir, opts, audio_info, spec.shape[1])

        self.tmp_db_data["spectrograms"].append(spec)
        self.tmp_db_data["infos"].append(audio_info)
        self.tmp_db_data["tags_df"].append(tags_df)
        self.tmp_db_data["tags_linear_presence"].append(tags_linear)
