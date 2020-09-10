import csv
import os
import pickle
import traceback
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage.interpolation import zoom

from .utils import load_annotations


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

    DB_TYPES = ["training", "validation"]

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

    @staticmethod
    def force_make_dir(dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath

    def get_dir(self, name, database=None):
        dir_path = None
        if database and dir_name in database:
            dir_path = database[dir_name]
        else:
            dir_path = self.default_paths.get(dir_name, "")
        return Path(dir_path)

    def get_option(self, name, database=None):
        option = None
        if database and name in database:
            option = database[name]
        else:
            option = self.default_options.get(name, "")
        if name.endswith("_dir"):
            option = Path(option)
        return option

    def get_default_paths(self):
        default_paths = {}
        data_opts = self.opts["data"]
        for path in self.DEFAULT_OPTIONS["paths"]:
            if path in data_opts:
                default_paths[path] = Path(data_opts[path])
        print(default_paths)
        return default_paths

    def get_default_options(self):
        data_opts = self.opts["data"]
        default_opts = {
            name: value for name, value in data_opts.items() if name != "databases"
        }
        return default_opts

    # def get_database_paths(self, database):
    #     database_paths = {}
    #     root_dir = None
    #     for path in self.PATHS:
    #         tmp_path = database.get(path, self.default_paths.get(path, ""))
    #         if root_dir:
    #             tmp_path = self.get_full_path(tmp_path, root_dir)
    #         else:
    #             root_dir = tmp_path
    #         database_paths[path] = tmp_path
    #     return database_paths

    def get_database_paths2(self, database):
        db_paths = {}
        root_dir = self.get_option("root_dir", database)
        db_paths["root_dir"] = root_dir

        for db_type in self.DB_TYPES:
            db_type_dir = self.get_full_path(
                self.get_option(db_type + "_dir", database), root_dir
            )
            db_paths[db_type + "_dir"] = db_type_dir
            db_paths[db_type + "_audio_dir"] = self.get_full_path(
                self.get_option("audio_dir", database), db_type_dir
            )
            db_paths[db_type + "_tags_dir"] = self.get_full_path(
                self.get_option("tags_dir", database), db_type_dir
            )
            db_paths[db_type + "_dest_dir"] = (
                self.get_full_path(self.get_option("dest_dir", database), db_type_dir)
                / database["name"]
            )

        return db_paths

    def generate_datasets(self):
        # iterate on databases
        #    load paths from config
        #        if train and validation paths provided, use them
        #        else split data using provided function (how?)
        #        generate file_lists
        #     generate training and validation subsets
        #        use file lists
        #        generate pickles
        pass

    def random_split(self, database, paths):
        return {"train": [], "validation": []}

    def check_file_lists(self, database, paths):
        return {"train": [], "validation": []}

    def create_dataset(self, database, split_funcs=None):
        paths = self.get_database_paths2(database)
        print(paths)

        # self.check_file_lists(database, paths)

        # Create training and validation subsets
        for db_type in self.DB_TYPES:
            # Get paths

            self.force_make_dir(paths[db_type + "_dest_dir"])

            file_list_path = paths[db_type + "_dest_dir"] / (
                "_".join([db_type, "file_list.csv"])
            )
            pkl_file_path = paths[db_type + "_dest_dir"] / (
                "_".join([db_type, "all_data.pkl"])
            )

            file_list = []

            # If file list for subset does not exists, create it
            if not file_list_path.exists():
                # Check if we have to split the original data
                if "validation_split" in database:
                    # Do it randomly if not function is provided
                    if not split_funcs or database["name"] not in split_funcs:
                        file_list = self.random_split(database, paths)[db_type]
                    else:
                        file_list = split_funcs[database["name"]](database, paths)[
                            db_type
                        ]
                else:
                    # Use db type audio directory
                    file_list = paths[db_type + "_audio_dir"].iterdir()
            else:
                with open(file_list_path, mode="r") as f:
                    reader = csv.reader(f)
                    for name in reader:
                        file_list.append(Path(name[0]))

            # TODO check overwrite
            if not pkl_file_path.exists() or database["overwrite"]:
                print("Generating dataset: ", database["name"])
                file_names_list = []
                tmp_vals = []

                for file_path in file_list:
                    if file_path.suffix.lower() != ".wav":
                        continue
                    try:
                        annots, wav, sample_rate = load_annotations(
                            file_path,
                            paths[db_type + "_tags_dir"],
                            self.opts["class_type"],
                        )
                        spec = self.generate_spectrogram(wav, sample_rate)

                        # reshape annotations
                        factor = float(spec.shape[1]) / annots.shape[0]
                        annots = zoom(annots, factor)

                        file_names_list.append(file_path)
                        tmp_vals.append((annots, spec))

                        if self.opts["data"].get("save_intermediates", False):
                            savename = (
                                paths["dest_dir"] / "intermediate" / file_path.name
                            ).with_suffix(".pkl")
                            # TODO: check overwrite
                            if not savename.exists() or self.opts.get(
                                "overwrite", False
                            ):
                                with open(savename, "wb") as f:
                                    pickle.dump((annots, spec), f, -1)
                    except Exception:
                        print("Error loading: " + str(file_path) + ", skipping.")
                        print(traceback.format_exc())

                # Save all data
                with open(pkl_file_path, "wb") as f:
                    pickle.dump(tmp_vals, f, -1)

                # Save file_list
                if not file_list_path.exists():
                    with open(file_list_path, mode="w") as f:
                        writer = csv.writer(f)
                        for name in file_names_list:
                            writer.writerow([name])

    def create_datasets(self, db_type="train"):
        for database in self.opts["data"]["databases"]:
            paths = self.get_database_paths2(database, db_type)

            self.force_make_dir(paths["dest_dir"])

            file_list_path = paths["dest_dir"] / ("_".join([db_type, "file_list.csv"]))
            pkl_file_path = paths["dest_dir"] / ("_".join([db_type, "all_data.pkl"]))
            print(file_list_path)
            print(pkl_file_path)

            if file_list_path.exists():
                print("Found file list")
            else:
                print("no file list found, generating one")
                file_names_list = []
                tmp_vals = []

                for file_path in paths["audio_dir"].iterdir():
                    if not file_path.suffix.lower() == ".wav":
                        continue
                    try:
                        annots, wav, sample_rate = load_annotations(
                            file_path, paths["tags_dir"], self.opts["class_type"]
                        )
                        spec = self.generate_spectrogram(wav, sample_rate)

                        # reshape annotations
                        factor = float(spec.shape[1]) / annots.shape[0]
                        annots = zoom(annots, factor)

                        file_names_list.append(file_path)
                        # tmp_vals.append((annots, spec))

                        if self.opts["data"].get("save_intermediates", False):
                            savename = (
                                paths["dest_dir"] / "intermediate" / file_path.name
                            ).with_suffix(".pkl")
                            if not savename.exists() or self.opts.get(
                                "overwrite", False
                            ):
                                with open(savename, "wb") as f:
                                    pickle.dump((annots, spec), f, -1)
                    except Exception:
                        print("Error loading: " + str(file_path) + ", skipping.")
                        print(traceback.format_exc())

                # Save all data
                with open(pkl_file_path, "wb") as f:
                    pickle.dump(tmp_vals, f, -1)

                # Save file_list
                print(file_names_list)
                with open(file_list_path, mode="w") as f:
                    writer = csv.writer(f)
                    for name in file_names_list:
                        writer.writerow([name])

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
