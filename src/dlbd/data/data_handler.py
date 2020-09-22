import csv
import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import zoom

from ..utils.file import list_files
from . import spectrogram
from . import tags


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

    def __init__(self, opts):
        self.opts = opts

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
        spec_folder = ""
        if use_spec_subfolder:
            spec_folder = spectrogram.get_spec_subfolder(self.opts["spectrogram"])

        paths["root"] = root_dir
        paths["audio"] = {"default": self.get_db_option("audio_dir", database)}
        paths["tags"] = {"default": self.get_db_option("tags_dir", database)}
        paths["dest"] = {
            "default": self.get_db_option("dest_dir", database) / spec_folder
        }
        paths["file_list"] = {}
        paths["pkl"] = {}

        for db_type in self.DB_TYPES:
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

    def check_file_lists(self, database, paths, split_funcs):
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
            if split_funcs and database["name"] in split_funcs:
                file_lists = split_funcs[database["name"]](paths, self, database)
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

    def generate_dataset(self, database, paths, file_list, db_type, overwrite):
        x, y = [], []
        print("Generating dataset: ", database["name"])

        class_type = self.get_db_option("class_type", database, "biotic")
        classes_file = self.get_db_option("classes_file", database, "classes.csv")

        classes_df = pd.read_csv(classes_file, skip_blank_lines=True)
        classes = classes_df.loc[classes_df.class_type == class_type].tag.values
        suffix = self.get_db_option("tags_suffix", database, "-sceneRect.csv")
        tags_with_audio = self.get_db_option("tags_with_audio", database, False)

        for file_path in file_list:
            try:

                annots, wav, sample_rate = tags.load_tags(
                    file_path,
                    paths["tags"][db_type],
                    suffix=suffix,
                    tags_with_audio=tags_with_audio,
                    classes=classes,
                )
                spec = spectrogram.generate_spectrogram(
                    wav, sample_rate, self.opts["spectrogram"]
                )

                # reshape annotations
                factor = float(spec.shape[1]) / annots.shape[0]
                annots = zoom(annots, factor)

                # file_names_list.append(file_path)
                x.append(spec)
                y.append(annots)

                if self.get_db_option("save_intermediates", database, False):
                    savename = (
                        paths["dest"][db_type] / "intermediate" / file_path.name
                    ).with_suffix(".pkl")
                    if not savename.exists() or overwrite:
                        with open(savename, "wb") as f:
                            pickle.dump((annots, spec), f, -1)
            except Exception:
                print("Error loading: " + str(file_path) + ", skipping.")
                print(traceback.format_exc())

        # Save all data
        if x and y:
            with open(paths["pkl"][db_type], "wb") as f:
                pickle.dump((x, y), f, -1)
                print("Saved file: ", paths["pkl"][db_type])
        return (x, y)

    def check_dataset(self, database, paths, file_list, db_type, load=False):
        # * Overwrite if generate_file_lists is true as file lists will be recreated
        overwrite = self.get_db_option(
            "overwrite", database, False
        ) or self.get_db_option("generate_file_lists", database, False)
        res = []
        if not paths["pkl"][db_type].exists() or overwrite:
            res = self.generate_dataset(database, paths, file_list, db_type, overwrite)
        elif load:
            with open(paths["pkl"][db_type], "rb") as f:
                res = pickle.load(f, -1)
                print("Loaded file: ", paths["pkl"][db_type])

        return res

    def check_datasets(self, split_funcs=None):
        for database in self.opts["databases"]:
            print("Checking database:", database["name"])
            paths = self.get_database_paths(database)
            file_lists = self.check_file_lists(database, paths, split_funcs)
            for db_type, file_list in file_lists.items():
                self.check_dataset(database, paths, file_list, db_type)

    # TODO : add spectrogram modification into the trainer, right before training/classifying
    def modify_spectrogram(self, spec):
        spec = np.log(self.opts["A"] + self.opts["B"] * spec)
        spec = spec - np.median(spec, axis=1, keepdims=True)
        return spec

    def load_file(self, file_name):
        specs, tags = pickle.load(open(file_name, "rb"))

        if not self.opts["learn_log"]:
            specs = [self.modify_spectrogram(spec) for spec in specs]

        return specs, tags

    def load_data(self, db_type):
        x_data = []
        y_data = []
        for database in self.opts["databases"]:
            print("Loading data for database:", database["name"])
            paths = self.get_database_paths(database)
            if not paths["pkl"][db_type].exists():
                raise ValueError(
                    "Database file not found. Please run check_datasets() before"
                )
            else:
                # x : spectrograms, y: tags
                x, y = self.load_file(paths["pkl"][db_type])
                x_data += x
                y_data += y

                # height = min(xx.shape[0] for xx in X_tmp)
                # X_tmp = [xx[-height:, :] for xx in X_tmp]
        return x_data, y_data
