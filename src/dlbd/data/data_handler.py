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

    PATHS = [
        "root_dir",
        "audio_dir",
        "tags_dir",
        "train_dir",
        "validation_dir",
        "dest_dir",
    ]

    def __init__(self, opts):
        self.opts = opts
        self.default_paths = self.get_default_paths()

    @staticmethod
    def get_full_path(path, root):
        path = Path(path)
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
        name = name + "_dir"
        dir_path = None
        if database:
            if name in database:
                dir_path = database[name + "_dir"]

        return dir_path

    def get_default_paths(self):
        default_paths = {}
        data_opts = self.opts["data"]
        for path in self.PATHS:
            if path in data_opts:
                default_paths[path] = Path(data_opts[path])
        return default_paths

    def get_database_paths(self, database):
        database_paths = {}
        root_dir = None
        for path in self.PATHS:
            tmp_path = database.get(path, self.default_paths.get(path, ""))
            if root_dir:
                tmp_path = self.get_full_path(tmp_path, root_dir)
            else:
                root_dir = tmp_path
            database_paths[path] = tmp_path
        return database_paths

    def create_datasets(self, data_type="train"):
        for database in self.opts["data"]["databases"]:
            paths = self.get_database_paths(database)

            dest_dir = paths["dest_dir"] / database["name"]
            self.force_make_dir(dest_dir)

            file_list_path = dest_dir / ("_".join([data_type, "file_list.csv"]))
            pkl_file_path = dest_dir / ("_".join([data_type, "all_data.pkl"]))
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
                        tmp_vals.append((annots, spec))

                        if self.opts["data"].get("save_intermediates", False):
                            savename = (
                                dest_dir / "intermediate" / file_path.name
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
                with open(file_names_list, mode="w") as f:
                    writer = csv.writer(f)
                    for name in file_names_list:
                        writer.writerow(name)

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
