import librosa
import pandas as pd
from scipy.ndimage.interpolation import zoom

from dlbd.lib.data_handler import DataHandler
from . import spectrogram, tag_manager


class AudioDataHandler(DataHandler):

    DEFAULT_OPTIONS = {
        "class_type": "biotic",
        "data_extensions": [".wav"],
        "classes_file": "classes.csv",
        "tags_suffix": "-sceneRect.csv",
    }

    DATA_STRUCTURE = {
        "spectrograms": [],
        "tags_df": [],
        "tags_linear_presence": [],
        "infos": [],
    }

    def get_spectrogram_subfolder_path(self, database):
        return spectrogram.get_spec_subfolder(
            self.load_option_group("spectrogram", database)
        )

    def merge_datasets(self, datasets):
        merged = super().merge_datasets(datasets)
        merged["tags_df"] = pd.concat(merged["tags_df"])
        return merged

    def finalize_dataset(self):
        self.tmp_db_data["tags_df"] = pd.concat(self.tmp_db_data["tags_df"])

    def load_data_options(self, database):
        opts = {}
        opts["tags"] = self.load_option_group("tags", database)
        opts["spectrogram"] = self.load_option_group("spectrogram", database)
        opts["classes"] = self.load_classes(database)
        return opts

    def load_file_data(self, file_path, tags_dir, opts):
        # load file and convert to spectrogram
        tag_opts = opts["tags"]
        spec_opts = opts["spectrogram"]
        sr = spec_opts.get("sample_rate", None)
        if sr and sr == "original":
            sr = None
        wav, sample_rate = librosa.load(str(file_path), sr=sr)

        audio_info = {
            "file_path": file_path,
            "sample_rate": sample_rate,
            "length": len(wav),
        }
        tag_df = tag_manager.get_tag_df(audio_info, tags_dir, tag_opts)

        # * NOTE: sp_opts can contain options not defined in spec_opts
        spec, sp_opts = spectrogram.generate_spectrogram(wav, sample_rate, spec_opts)

        audio_info["spec_opts"] = sp_opts
        tmp_tags = tag_manager.filter_classes(tag_df, opts["classes"])
        tag_presence = tag_manager.get_tag_presence(tmp_tags, audio_info, tag_opts)
        factor = float(spec.shape[1]) / tag_presence.shape[0]
        zoomed_presence = zoom(tag_presence, factor)

        self.tmp_db_data["spectrograms"].append(spec)
        self.tmp_db_data["infos"].append(audio_info)
        self.tmp_db_data["tags_df"].append(tmp_tags)
        self.tmp_db_data["tags_linear_presence"].append(zoomed_presence)
