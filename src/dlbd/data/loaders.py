import traceback
from pathlib import Path

import pandas as pd
from mouffet.data.data_loader import DataLoader

from . import audio_utils, tag_utils


class AudioDataLoader(DataLoader):

    CALLBACKS = {"onload": {"tags_df": tag_utils.prepare_tags}}

    def dataset_options(self, database):
        opts = {}
        opts["tags"] = database.tags
        opts["spectrogram"] = database.spectrogram
        opts["classes"] = self.load_classes(database)
        return opts

    def load_file_data(self, file_path, tags_dir, opts):
        spec, audio_info = audio_utils.load_audio_data(file_path, opts["spectrogram"])
        tags_df, tags_linear = tag_utils.load_tags(
            tags_dir, opts, audio_info, spec.shape[1]
        )
        self.data["spectrograms"].append(spec)
        self.data["infos"].append(audio_info)
        self.data["tags_df"].append(tags_df)
        self.data["tags_linear_presence"].append(tags_linear)

    def finalize_dataset(self):
        self.data["tags_df"] = pd.concat(self.data["tags_df"])


class BADChallengeDataLoader(AudioDataLoader):

    CALLBACKS = {}

    def finalize_dataset(self):
        pass

    def load_file_data(self, file_path, opts):
        spec, audio_info = audio_utils.load_audio_data(file_path, opts["spectrogram"])
        self.data["spectrograms"].append(spec)
        self.data["infos"].append(audio_info)

    def generate_dataset(self, database, paths, file_list, db_type, overwrite):
        db_opts = self.dataset_options(database)
        tags_dir = paths["tags"][db_type]
        self.data["tags_df"] = tag_utils.get_bad_challenge_tag_df(tags_dir)
        cpt = 0
        for file_path in file_list:
            if cpt == 10:
                return
            try:
                if not isinstance(file_path, Path):
                    file_path = Path(file_path)
                self.load_file_data(file_path=file_path, opts=db_opts)
                cpt += 1
            except Exception:
                print("Error loading: " + str(file_path) + ", skipping.")
                print(traceback.format_exc())
                self.data = None
