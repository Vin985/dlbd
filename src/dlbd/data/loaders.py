import traceback
from pathlib import Path

import pandas as pd
from mouffet.data import DataLoader
from mouffet import common_utils

from . import audio_utils, tag_utils


class AudioDataLoader(DataLoader):

    CALLBACKS = {"onload": {"tags_df": tag_utils.prepare_tags}}

    def dataset_options(self, database):
        opts = {}
        opts["tags"] = database.tags
        opts["spectrogram"] = database.spectrogram
        opts["classes"] = self.load_classes(database)
        opts["reference_classes"] = self.load_reference_classes(database)
        return opts

    def load_file_data(self, file_path, tags_dir, opts, missing=None):
        load_audio = common_utils.any_in_list(
            missing, ["spectrograms", "metadata", "spec_opts", "tags_linear_presence"]
        )
        load_tags = common_utils.any_in_list(missing, ["tags_df"])

        if load_audio:
            spec, metadata, spec_opts = audio_utils.load_audio_data(
                file_path, opts["spectrogram"]
            )
            self.data["spectrograms"].append(spec)
            self.data["metadata"].append(metadata)
            self.data["spec_opts"].append(spec_opts)
        if load_tags:
            tags_df = tag_utils.load_tags_df(tags_dir, opts, metadata)

            self.data["tags_df"].append(tags_df)
            if load_audio:
                tags_linear = tag_utils.load_tags_presence(
                    tags_df, opts, metadata, spec.shape[1]
                )
                self.data["tags_linear_presence"].append(tags_linear)

    def finalize_dataset(self):
        self.data["tags_df"] = pd.concat(self.data["tags_df"])

    def load_classes(self, database):
        class_type = database.class_type
        classes_file = database.classes_file

        classes_df = pd.read_csv(classes_file, skip_blank_lines=True)
        classes = (
            classes_df.loc[
                classes_df["class_type"]  # pylint: disable=unsubscriptable-object
                == class_type
            ]
            .tag.str.lower()
            .values
        )
        return classes

    def load_reference_classes(self, database):
        ref_classes_file = database.reference_classes_file
        classes_df = pd.read_csv(ref_classes_file, skip_blank_lines=True)
        return classes_df


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
        # cpt = 0
        for file_path in file_list:
            # if cpt == 10:
            #     return
            try:
                if not isinstance(file_path, Path):
                    file_path = Path(file_path)
                self.load_file_data(file_path=file_path, opts=db_opts)
                # cpt += 1
            except Exception:
                print("Error loading: " + str(file_path) + ", skipping.")
                print(traceback.format_exc())
                self.data = None
