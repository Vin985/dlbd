import csv
import os
import pickle
import random
import traceback
from pathlib import Path

import librosa
import numpy as np
from librosa.core import audio
from scipy.ndimage.interpolation import zoom

from .utils import load_annotations
from ..utils.file import list_files


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
        # self.default_paths = self.get_default_paths()
        self.default_options = self.get_default_options()

    @staticmethod
    def get_full_path(path, root):
        if path.is_absolute():
            return path
        else:
            return root / path

    def get_option(self, name, database=None, default=""):
        option = None
        if database and name in database:
            option = database[name]
        else:
            option = self.default_options.get(name, default)
        if name.endswith("_dir"):
            option = Path(option)
        return option

    def get_default_options(self):
        data_opts = self.opts["data"]
        default_opts = {
            name: value for name, value in data_opts.items() if name != "databases"
        }
        return default_opts

    def get_database_paths(self, database):
        paths = {}
        root_dir = self.get_option("root_dir", database)
        paths["root"] = root_dir
        paths["audio"] = {"default": self.get_option("audio_dir", database)}
        paths["tags"] = {"default": self.get_option("tags_dir", database)}
        paths["dest"] = {"default": self.get_option("dest_dir", database)}
        paths["file_list"] = {}
        paths["pkl"] = {}

        for db_type in self.DB_TYPES:
            db_type_dir = self.get_full_path(
                self.get_option(db_type + "_dir", database), root_dir
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
                self.get_option("audio_ext", database, [".wav"]),
                self.get_option("recursive", database, False),
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
        if not file_lists_exist or self.get_option(
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

    # def create_datasets(self, database, split_funcs=None):
    #     for database in self.opts["data"]["databases"]:
    #         paths = self.get_database_paths(database)
    #         for db_type in self.DB_TYPES:
    #             self.create_dataset(database, paths, db_type, split_funcs)

    def check_dataset(self, database, paths, file_list, db_type):
        # * Overwrite if generate_file_lists is true as file lists will be recreated
        overwrite = self.get_option("overwrite", database, False) or self.get_option(
            "generate_file_lists", database, False
        )
        if not paths["pkl"][db_type].exists() or overwrite:
            print("Generating dataset: ", database["name"])
            tmp_vals = []

            for file_path in file_list:
                try:
                    annots, wav, sample_rate = load_annotations(
                        file_path,
                        paths["tags"][db_type],
                        self.get_option("tags_suffix", database, "-sceneRect.csv"),
                        self.get_option("tags_with_audio", database, False),
                        self.get_option("class_type", database, "biotic"),
                    )
                    spec = self.generate_spectrogram(wav, sample_rate)

                    # reshape annotations
                    factor = float(spec.shape[1]) / annots.shape[0]
                    annots = zoom(annots, factor)

                    # file_names_list.append(file_path)
                    tmp_vals.append((annots, spec))

                    if self.get_option("save_intermediates", database, False):
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
            if tmp_vals:
                with open(paths["pkl"][db_type], "wb") as f:
                    pickle.dump(tmp_vals, f, -1)
                    print("Saved file: ", paths["pkl"][db_type])

    def check_datasets(self, split_funcs=None):
        for database in self.opts["data"]["databases"]:
            print("Checking database:", database["name"])
            paths = self.get_database_paths(database)
            file_lists = self.check_file_lists(database, paths, split_funcs)
            for db_type, file_list in file_lists.items():
                print(db_type)
                self.check_dataset(database, paths, file_list, db_type)

    # def create_dataset(self, database, paths, db_type, split_funcs=None):
    #     # Create destination paths
    #     self.force_make_dir(paths[db_type + "_dest_dir"])
    #     file_list_path = paths[db_type + "_dest_dir"] / (
    #         "_".join([db_type, "file_list.csv"])
    #     )
    #     pkl_file_path = paths[db_type + "_dest_dir"] / (
    #         "_".join([db_type, "all_data.pkl"])
    #     )

    #     file_list = []
    #     # * If file list for subset does not exists, create it
    #     if not file_list_path.exists():
    #         validation_split = self.get_option("validation_split", database)
    #         # * Check if we have to split the original data
    #         if validation_split:
    #             if not split_funcs or database["name"] not in split_funcs:
    #                 # * Do it randomly if not function is provided
    #                 file_list = self.random_split(validation_split, paths)[db_type]
    #             else:
    #                 # * else use the provided splitting function
    #                 file_list = split_funcs[database["name"]](
    #                     validation_split, paths, self, database
    #                 )[db_type]
    #         else:
    #             # * Use db type audio directory
    #             file_list = paths[db_type + "_audio_dir"].iterdir()
    #     else:
    #         with open(file_list_path, mode="r") as f:
    #             reader = csv.reader(f)
    #             for name in reader:
    #                 file_list.append(Path(name[0]))

    #     # TODO check overwrite
    #     if not pkl_file_path.exists() or database["overwrite"]:
    #         print("Generating dataset: ", database["name"])
    #         tmp_vals = []

    #         for file_path in file_list:
    #             try:
    #                 annots, wav, sample_rate = load_annotations(
    #                     file_path,
    #                     paths[db_type + "_tags_dir"],
    #                     self.get_option("tags_suffix", database, "-sceneRect.csv"),
    #                     self.get_option("tags_with_audio", database, False),
    #                     self.get_option("class_type", database, "biotic"),
    #                 )
    #                 spec = self.generate_spectrogram(wav, sample_rate)

    #                 # reshape annotations
    #                 factor = float(spec.shape[1]) / annots.shape[0]
    #                 annots = zoom(annots, factor)

    #                 # file_names_list.append(file_path)
    #                 tmp_vals.append((annots, spec))

    #                 if self.opts["data"].get("save_intermediates", False):
    #                     savename = (
    #                         paths["dest_dir"] / "intermediate" / file_path.name
    #                     ).with_suffix(".pkl")
    #                     # TODO: check overwrite
    #                     if not savename.exists() or self.opts.get("overwrite", False):
    #                         with open(savename, "wb") as f:
    #                             pickle.dump((annots, spec), f, -1)
    #             except Exception:
    #                 print("Error loading: " + str(file_path) + ", skipping.")
    #                 print(traceback.format_exc())

    #         # Save all data
    #         if tmp_vals:
    #             with open(pkl_file_path, "wb") as f:
    #                 pickle.dump(tmp_vals, f, -1)

    #         # # Save file_list
    #         # if not file_list_path.exists() and file_names_list:
    #         #     with open(file_list_path, mode="w") as f:
    #         #         writer = csv.writer(f)
    #         #         for name in file_names_list:
    #         #             writer.writerow([name])

    def generate_spectrogram(self, wav, sample_rate):

        if self.opts["spec_type"] == "mel":
            spec = librosa.feature.melspectrogram(
                wav,
                sr=sample_rate,
                n_fft=self.opts.get("n_fft", 2048),
                hop_length=self.opts.get("hop_length", 1024),
                n_mels=self.opts.get("n_mels", 32),
            )
            spec = spec.astype(np.float32)
        else:
            raise AttributeError("No other spectrogram supported yet")
        return spec

    def load_file(self, file_name):

        annots, spec = pickle.load(open(file_name, "rb"))
        annots = annots[self.opts["classname"]]
        # reshape annotations
        # factor = float(spec.shape[1]) / annots.shape[0]
        # annots = zoom(annots, factor)
        # create sampler
        if not self.opts["learn_log"]:
            spec = np.log(self.opts["A"] + self.opts["B"] * spec)
            spec = spec - np.median(spec, axis=1, keepdims=True)

        return annots, spec

    def load_data(self, data_type="train"):
        # load data and make list of specsamplers
        X = []
        y = []

        for root_dir in self.opts["root_dirs"]:
            X_tmp = []
            y_tmp = []
            src_dir = (
                Path(root_dir)
                / self.opts[data_type + "_dir"]
                / self.opts["dest_dir"]
                / self.opts["spec_type"]
            )
            all_path = Path(src_dir / "all.pkl")
            if all_path.exists():
                X_tmp, y_tmp = pickle.load(open(all_path, "rb"))

            else:
                for file_name in os.listdir(src_dir):
                    print("Loading file: ", file_name)
                    annots, spec = self.load_file(src_dir / file_name)
                    X_tmp.append(spec)
                    y_tmp.append(annots)

                height = min(xx.shape[0] for xx in X_tmp)
                X_tmp = [xx[-height:, :] for xx in X_tmp]

                with open(all_path, "wb") as f:
                    pickle.dump((X_tmp, y_tmp), f, -1)

            X += X_tmp
            y += y_tmp
        return X, y
