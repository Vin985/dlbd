import csv
import pickle
import traceback
from pathlib import Path
import feather

import librosa
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import zoom

from ..utils.file import list_files
from . import spectrogram
from . import tag_manager


class DataHandler:

    DEFAULT_OPTIONS2 = {
        "paths": [
            "root_dir",
            "audio_dir",
            "tags_dir",
            "training_dir",
            "validation_dir",
            "dest_dir",
        ],
        "options": ["overwrite"],
    }

    DEFAULT_OPTIONS = [
        "root_dir",
        "audio_dir",
        "tags_dir",
        "training_dir",
        "validation_dir",
        "dest_dir",
        "overwrite",
    ]

    DB_TYPES = ["test", "training", "validation"]

    def __init__(self, opts, split_funcs=None):
        self.opts = opts
        self.split_funcs = split_funcs
        print(split_funcs)

    @staticmethod
    def get_full_path(path, root):
        if path.is_absolute():
            return path
        else:
            return root / path

    def get_db_option(self, name, database=None, default=""):
        option = None
        if database and name in database:
            option = database[name]
        else:
            option = self.opts.get(name, default)
        if name.endswith("_dir"):
            option = Path(option)
        return option

    def get_database_paths(self, database):
        paths = {}
        root_dir = self.get_db_option("root_dir", database)

        use_spec_subfolder = self.get_db_option("use_spec_subfolder", database, True)
        use_class_subfolder = self.get_db_option("use_class_subfolder", database, True)
        spec_folder = (
            spectrogram.get_spec_subfolder(self.opts["spectrogram"])
            if use_spec_subfolder
            else ""
        )
        class_folder = (
            self.get_db_option("class_type", database, "biotic")
            if use_class_subfolder
            else ""
        )

        paths["root"] = root_dir
        paths["audio"] = {"default": self.get_db_option("audio_dir", database)}
        paths["tags"] = {"default": self.get_db_option("tags_dir", database)}
        paths["dest"] = {
            "default": self.get_db_option("dest_dir", database)
            / class_folder
            / spec_folder
        }
        paths["file_list"] = {}
        paths["pkl"] = {}
        paths["tag_df"] = {}

        for db_type in self.get_db_option("db_types", database, self.DB_TYPES):
            db_type_dir = self.get_full_path(
                self.get_db_option(db_type + "_dir", database), root_dir
            )
            paths[db_type + "_dir"] = db_type_dir
            paths["audio"][db_type] = self.get_full_path(
                paths["audio"]["default"], db_type_dir
            )
            paths["tags"][db_type] = self.get_full_path(
                paths["tags"]["default"], db_type_dir
            )
            dest_dir = (
                self.get_full_path(paths["dest"]["default"], db_type_dir)
                / database["name"]
            )
            paths["dest"][db_type] = dest_dir
            paths["file_list"][db_type] = dest_dir / (db_type + "_file_list.csv")
            paths["pkl"][db_type] = dest_dir / (db_type + "_data.pkl")
            paths["tag_df"][db_type] = dest_dir / (db_type + "_tags.feather")

        return paths

    @staticmethod
    def save_file_list(db_type, file_list, paths):
        file_list_path = paths["dest"][db_type] / (db_type + "_file_list.csv")
        file_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_list_path, mode="w") as f:
            writer = csv.writer(f)
            for name in file_list:
                writer.writerow([name])
            print("Saved file list:", str(file_list_path))

    def get_audio_file_lists(self, paths, database):
        res = {}
        for db_type in self.DB_TYPES:
            res[db_type] = list_files(
                paths["audio"][db_type],
                self.get_db_option("audio_ext", database, [".wav"]),
                self.get_db_option("recursive", database, False),
            )
        return res

    @staticmethod
    def load_file_lists(paths):
        res = {}
        for db_type, path in paths["file_list"].items():
            file_list = []
            with open(path, mode="r") as f:
                reader = csv.reader(f)
                for name in reader:
                    file_list.append(Path(name[0]))
            res[db_type] = file_list
            print("Loaded file: " + str(path))
        return res

    def check_file_lists(self, database, paths):
        file_lists = {}
        msg = "Checking file lists for database {0}... ".format(database["name"])
        file_lists_exist = all([path.exists() for path in paths["file_list"].values()])
        # * Check if file lists are missing or need to be regenerated
        if not file_lists_exist or self.get_db_option(
            "generate_file_lists", database, False
        ):
            print(msg + "Generating file lists...")
            file_lists = {}
            # * Check if we have a dedicated function to split the original data
            if self.split_funcs and database["name"] in self.split_funcs:
                file_lists = self.split_funcs[database["name"]](paths, self, database)
            else:
                file_lists = self.get_audio_file_lists(paths, database)
            # * Save results
            for db_type, file_list in file_lists.items():
                self.save_file_list(db_type, file_list, paths)
        else:
            # * Load files
            print(msg + "Found all file lists. Now loading...")
            file_lists = self.load_file_lists(paths)
        return file_lists

    def load_classes(self, database):
        class_type = self.get_db_option("class_type", database, "biotic")
        classes_file = self.get_db_option("classes_file", database, "classes.csv")

        classes_df = pd.read_csv(classes_file, skip_blank_lines=True)
        classes = (
            classes_df.loc[classes_df.class_type == class_type].tag.str.lower().values
        )
        return classes

    def load_tags_opts(self, database, db_type):
        tags_opts = {
            "suffix": self.get_db_option("tags_suffix", database, "-sceneRect.csv"),
            "tags_with_audio": self.get_db_option("tags_with_audio", database, False),
            "classes": self.load_classes(database),
            "columns": self.get_db_option("tags_columns", database, None),
            "columns_type": self.get_db_option("tags_columns_type", database, None),
        }
        return tags_opts

    def generate_dataset(self, database, paths, file_list, db_type, overwrite):
        spectrograms, tags_df, training_tags, infos = [], [], [], []
        print("Generating dataset: ", database["name"])

        tags_opts = self.load_tags_opts(database, db_type)

        for file_path in file_list:
            try:
                # load file and convert to spectrogram
                wav, sample_rate = librosa.load(
                    str(file_path), self.opts["spectrogram"].get("sample_rate", None)
                )

                audio_info = {
                    "file_path": file_path,
                    "sample_rate": sample_rate,
                    "length": len(wav),
                }
                tag_df = tag_manager.get_tag_df(
                    audio_info, paths["tags"][db_type], tags_opts
                )

                spec, opts = spectrogram.generate_spectrogram(
                    wav, sample_rate, self.opts["spectrogram"]
                )

                audio_info["spec_opts"] = opts
                if not db_type == "test":
                    tag_presence = tag_manager.get_tag_presence(
                        tag_df, audio_info, tags_opts
                    )
                    factor = float(spec.shape[1]) / tag_presence.shape[0]
                    train_tags = zoom(tag_presence, factor)
                else:
                    train_tags = []

                # file_names_list.append(file_path)
                spectrograms.append(spec)
                tags_df.append(tag_df)
                training_tags.append(train_tags)
                infos.append(audio_info)

                if self.get_db_option("save_intermediates", database, False):
                    savename = (
                        paths["dest"][db_type] / "intermediate" / file_path.name
                    ).with_suffix(".pkl")
                    if not savename.exists() or overwrite:
                        with open(savename, "wb") as f:
                            pickle.dump((spec, training_tags), f, -1)
            except Exception:
                print("Error loading: " + str(file_path) + ", skipping.")
                print(traceback.format_exc())

        # Save all data
        if spectrograms and tags_df:
            with open(paths["pkl"][db_type], "wb") as f:
                pickle.dump((spectrograms, training_tags, infos), f, -1)
                print("Saved file: ", paths["pkl"][db_type])
            tags_df = pd.concat(tags_df)
            print(tags_df)
            feather.write_dataframe(tags_df, str(paths["tag_df"][db_type]))
        return (spectrograms, training_tags, infos)

    def check_dataset(self, database, paths, file_list, db_type, load=False):
        # * Overwrite if generate_file_lists is true as file lists will be recreated
        overwrite = self.get_db_option(
            "overwrite", database, False
        ) or self.get_db_option("generate_file_lists", database, False)
        res = []
        if (
            not paths["pkl"][db_type].exists()
            or not paths["tag_df"][db_type].exists()
            or overwrite
        ):
            res = self.generate_dataset(database, paths, file_list, db_type, overwrite)
        elif load:
            with open(paths["pkl"][db_type], "rb") as f:
                res = pickle.load(f, -1)
                print("Loaded file: ", paths["pkl"][db_type])

        return res

    def check_datasets(self):
        for database in self.opts["databases"]:
            print("Checking database:", database["name"])
            paths = self.get_database_paths(database)
            file_lists = self.check_file_lists(database, paths)
            for db_type, file_list in file_lists.items():
                self.check_dataset(database, paths, file_list, db_type)

    # TODO : add spectrogram modification into the trainer, right before training/classifying
    def modify_spectrogram(self, spec):
        spec = np.log(self.opts["model"]["A"] + self.opts["model"]["B"] * spec)
        spec = spec - np.median(spec, axis=1, keepdims=True)
        return spec

    def load_file(self, file_name):
        return pickle.load(open(file_name, "rb"))

    def load_data(self, db_type, by_dataset=False):
        spectrograms = []
        annotations = []
        infos = []
        res = {}
        for database in self.opts["databases"]:
            if db_type in self.get_db_option("db_types", database, self.DB_TYPES):
                print(
                    "Loading {0} data for database: {1}".format(
                        db_type, database["name"]
                    )
                )
                paths = self.get_database_paths(database)
                if not paths["pkl"][db_type].exists():
                    raise ValueError(
                        "Database file not found. Please run check_datasets() before"
                    )
                else:
                    # x : spectrograms, y: tags
                    specs, annots, info = self.load_file(paths["pkl"][db_type])
                    if by_dataset:
                        res[database["name"]] = (specs, annots, info)
                    else:
                        spectrograms += specs
                        annotations += annots
                        infos += info

                    # height = min(xx.shape[0] for xx in X_tmp)
                    # X_tmp = [xx[-height:, :] for xx in X_tmp]
        res = res or (spectrograms, annotations, infos)
        return res
