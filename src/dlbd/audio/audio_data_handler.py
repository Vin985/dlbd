from scipy.ndimage.interpolation import zoom
from ..lib.data_handler import DataHandler
from . import spectrogram
from . import tag_manager
import librosa


class AudioDataHandler(DataHandler):

    DEFAULT_OPTIONS = {
        "class_type": "biotic",
        "data_extensions": [".wav"],
        "classes_file": "classes.csv",
        "tags_suffix": "-sceneRect.csv",
    }

    DATA_STRUCTURE = {
        "data": [],
        "tags": {"df": [], "linear_presence": [], "linear_index": []},
        "infos": [],
    }

    def get_spectrogram_subfolder_path(self, database):
        return spectrogram.get_spec_subfolder(self.opts["spectrogram"])

    def load_file_data(self, file_path, tags_dir, tag_opts, db_type):
        # load file and convert to spectrogram
        wav, sample_rate = librosa.load(
            str(file_path), self.opts["spectrogram"].get("sample_rate", None)
        )

        audio_info = {
            "file_path": file_path,
            "sample_rate": sample_rate,
            "length": len(wav),
        }
        tag_df = tag_manager.get_tag_df(audio_info, tags_dir, tag_opts)

        # TODO: allow spectrogram options override
        spec, opts = spectrogram.generate_spectrogram(
            wav, sample_rate, self.opts["spectrogram"]
        )

        audio_info["spec_opts"] = opts
        if not db_type == "test":
            tmp_tags = tag_manager.filter_classes(tag_df, tag_opts["classes"])
            tag_presence = tag_manager.get_tag_presence(tmp_tags, audio_info, tag_opts)
            factor = float(spec.shape[1]) / tag_presence.shape[0]
            train_tags = zoom(tag_presence, factor)
        else:
            train_tags = []

        return None
