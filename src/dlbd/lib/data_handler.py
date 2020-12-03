import csv
import pickle
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

import feather
import numpy as np
from numpy.lib.npyio import load
import pandas as pd

from ..utils.file import ensure_path_exists, get_full_path, list_files

"""
A class that handles all data related business. While this class provides convenience functions,
this should be subclassed
"""


class DataHandler(ABC):

    DB_TYPES = ["test", "training", "validation"]

    DEFAULT_OPTIONS = {
        "class_type": "",
        "data_extensions": [""],
        "classes_file": "classes.csv",
        "tags_suffix": "-sceneRect.csv",
    }

    DATA_STRUCTURE = {"data": [], "tags": []}

    def __init__(self, opts, split_funcs=None):
        self.opts = opts
        self.split_funcs = split_funcs
        self.tmp_db_data = None

    def load_option_group(self, group, database=None):
        opts = self.opts[group]
        if database and group in database:
            db_opts = database[group]
            opts.update(db_opts)
        return opts

    def get_db_option(self, name, database=None, default="", group=None):
        """Get an option for the selected database. When the option is not present in the database,
        the default section is checked. If nothing is found, the 'default' argument is used.

        Args:
            name (str): The name of the option to look for
            database (dict, optional): An optional dictionary holding the options for the
            specified database. Defaults to None.
            default (str, optional): The default value to return in case the option is found
            neither in the database options nor in the default options. Defaults to "".

        Returns:
            object or Path: returns the object corresponding to the option as returned from pyyaml.
            If the option is a path and the 'name' argument ends with '_dir' or '_path', a
            pathlib.Path object is returned.
        """
        option = None

        # db = self.opts
        # if database:
        #     if group:
        #         if group in database:
        #             db = database[group]
        #         else:
        #             db = self.opts[group]
        #     elif name in database:
        #         db = database
        # option = db.get(name, default)
        # print(name, option)

        if database and name in database:
            option = database[name]
        else:
            option = self.opts.get(name, default)
        if isinstance(name, str) and (name.endswith("_dir") or name.endswith("_path")):
            option = Path(option)
        return option

    def get_class_subfolder_path(self, database):
        """Default implementation for a class subfolder

        Args:
            database (dict): The dictionary holding all option for the specific database.

        Returns:
            str: The class name
        """
        return self.get_db_option(
            "class_type", database, self.DEFAULT_OPTIONS["class_type"]
        )

    def get_subfolders(self, database):
        """Generate subfolders based on a list provided in the 'use_subfolders' option.
        For each item in the list, this function will try to call the 
        get_itemname_folder_path(database) method from the DataHandler instance, where itemname is
        the name of the current item in the list. For example, if the item is "class", then the
        function will attempt to call the 'get_class_folder_path' method. 
        If the method is not found, the option is skipped.
        Note that the called function should have the following signature:
        get_itemname_folder_path(database) -> str or pathlib.Path

        Args:
            database (dict): The dictionary holding all option for the specific database.

        Returns:
            pathlib.Path: a Path
        """
        res = Path("")
        subfolders = self.get_db_option("use_subfolders", database, None)
        if subfolders:
            if isinstance(subfolders, str):
                subfolders = [subfolders]
            for subfolder_type in subfolders:
                func_name = "_".join(["get", subfolder_type, "subfolder_path"])
                print(func_name)
                if hasattr(self, func_name) and callable(getattr(self, func_name)):
                    res /= getattr(self, func_name)(database)
                else:
                    print(
                        "Warning! No function found for getting the subfolder path for the '"
                        + subfolder_type
                        + "' option. Check if this is the correct value in the "
                        + "'use_subfolders' option or create a '"
                        + func_name
                        + "' function in your DataHandler instance."
                    )
        return res

    def get_save_dest_paths(self, dest_dir, db_type, subfolders):
        """Create

        Args:
            dest_dir ([type]): [description]
            db_type ([type]): [description]
            subfolders ([type]): [description]

        Returns:
            [type]: [description]
        """
        res = {}
        for key in self.DATA_STRUCTURE:
            ext = "feather" if key.endswith("_df") else "pkl"
            res[key] = dest_dir / subfolders / (db_type + "_" + key + "." + ext)
        return res

    def get_database_paths(self, database):
        paths = {}
        root_dir = self.get_db_option("root_dir", database)

        subfolders = self.get_subfolders(database)

        paths["root"] = root_dir
        paths["data"] = {"default": self.get_db_option("data_dir", database)}
        paths["tags"] = {"default": self.get_db_option("tags_dir", database)}
        paths["dest"] = {
            "default": self.get_db_option("dest_dir", database) / database["name"]
        }
        paths["file_list"] = {}
        paths["save_dests"] = {}

        for db_type in self.get_db_option("db_types", database, self.DB_TYPES):
            db_type_dir = get_full_path(
                self.get_db_option(db_type + "_dir", database), root_dir
            )
            paths[db_type + "_dir"] = db_type_dir
            paths["data"][db_type] = get_full_path(
                paths["data"]["default"], db_type_dir
            )
            paths["tags"][db_type] = get_full_path(
                paths["tags"]["default"], db_type_dir
            )
            dest_dir = get_full_path(paths["dest"]["default"], db_type_dir)
            paths["dest"][db_type] = dest_dir
            paths["file_list"][db_type] = self.get_db_option(
                db_type + "_file_list_path",
                database,
                dest_dir / (db_type + "_file_list.csv"),
            )
            paths["save_dests"][db_type] = self.get_save_dest_paths(
                dest_dir, db_type, subfolders
            )
            # paths["pkl"][db_type] = dest_dir / subfolders / (db_type + "_data.pkl")
            # paths["tag_df"][db_type] = dest_dir / (db_type + "_tags.feather")
        return paths

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

    @staticmethod
    def save_file_list(db_type, file_list, paths):
        file_list_path = paths["dest"][db_type] / (db_type + "_file_list.csv")
        with open(ensure_path_exists(file_list_path, is_file=True), mode="w") as f:
            writer = csv.writer(f)
            for name in file_list:
                writer.writerow([name])
            print("Saved file list:", str(file_list_path))

    def get_data_file_lists(self, paths, database):
        res = {}
        for db_type in self.get_db_option("db_types", database, self.DB_TYPES):
            res[db_type] = list_files(
                paths["data"][db_type],
                self.get_db_option(
                    "data_ext", database, [self.DEFAULT_OPTIONS["data_extensions"]]
                ),
                self.get_db_option("recursive", database, False),
            )
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
                file_lists = self.get_data_file_lists(paths, database)
            # * Save results
            for db_type, file_list in file_lists.items():
                self.save_file_list(db_type, file_list, paths)
        else:
            # * Load files
            print(msg + "Found all file lists. Now loading...")
            file_lists = self.load_file_lists(paths)
        return file_lists

    def load_classes(self, database):
        class_type = self.get_db_option(
            "class_type", database, self.DEFAULT_OPTIONS["class_type"]
        )
        classes_file = self.get_db_option(
            "classes_file", database, self.DEFAULT_OPTIONS["classes_file"]
        )

        classes_df = pd.read_csv(classes_file, skip_blank_lines=True)
        classes = (
            classes_df.loc[classes_df.class_type == class_type].tag.str.lower().values
        )
        return classes

    @abstractmethod
    def load_file_data(self, file_path, tags_dir, opts):
        data, tags = [], []
        return data, tags

    def save_dataset(self, paths, db_type):
        if self.tmp_db_data:
            for key, value in self.tmp_db_data.items():
                path = paths["save_dests"][db_type][key]
                print(path)
                if path.suffix == ".pkl":
                    with open(ensure_path_exists(path, is_file=True), "wb") as f:
                        pickle.dump(value, f, -1)
                        print("Saved file: ", path)
                elif path.suffix == ".feather":
                    value = value.reset_index(drop=True)
                    feather.write_dataframe(value, path)

    def finalize_dataset(self):
        """ Callback function called after data generation is finished but before it is saved
        in case some further action must be done after all files are loaded
        (e.g. dataframe concatenation)
        """
        pass

    @abstractmethod
    def load_data_options(self, database):
        return {}

    def generate_dataset(self, database, paths, file_list, db_type, overwrite):
        self.tmp_db_data = deepcopy(self.DATA_STRUCTURE)
        print("Generating dataset: ", database["name"])

        data_opts = self.load_data_options(database)

        for file_path in file_list:
            try:
                intermediate = self.load_file_data(
                    file_path=file_path, tags_dir=paths["tags"][db_type], opts=data_opts
                )

                if self.get_db_option("save_intermediates", database, False):
                    savename = (
                        paths["dest"][db_type] / "intermediate" / file_path.name
                    ).with_suffix(".pkl")
                    if not savename.exists() or overwrite:
                        with open(savename, "wb") as f:
                            pickle.dump(intermediate, f, -1)
            except Exception:
                print("Error loading: " + str(file_path) + ", skipping.")
                print(traceback.format_exc())
                self.tmp_db_data = None
        self.finalize_dataset()
        # Save all data
        self.save_dataset(paths, db_type)
        self.tmp_db_data = None

    def check_dataset_exists(self, paths, db_type):
        for key in self.DATA_STRUCTURE:
            if not paths["save_dests"][db_type][key].exists():
                return False
        return True

    def check_dataset(self, database, paths, file_list, db_type):
        # * Overwrite if generate_file_lists is true as file lists will be recreated
        overwrite = self.get_db_option(
            "overwrite", database, False
        ) or self.get_db_option("generate_file_lists", database, False)
        if not self.check_dataset_exists(paths, db_type) or overwrite:
            self.generate_dataset(database, paths, file_list, db_type, overwrite)
        # elif load:
        #     with open(paths["pkl"][db_type], "rb") as f:
        #         res = pickle.load(f, -1)
        #         print("Loaded file: ", paths["pkl"][db_type])

    def check_datasets(self, databases=None):
        if not databases:
            databases = self.opts["databases"]
        for database in databases:
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
        if file_name.suffix == ".feather":
            return feather.read_dataframe(str(file_name))
        else:
            return pickle.load(open(file_name, "rb"))

    def merge_datasets(self, datasets):
        merged = deepcopy(self.DATA_STRUCTURE)
        for dataset in datasets.values():
            for key in merged:
                merged[key] += dataset[key]
        return merged

    def get_file_types(self, load_opts):
        file_types = load_opts.get("file_types", "all")
        if file_types == "all":
            file_types = self.DATA_STRUCTURE.keys()
        else:
            # * Make sure we only have valid keys
            file_types = [ft for ft in file_types if ft in self.DATA_STRUCTURE]
        return file_types

    def load_dataset(self, database, db_type, load_opts=None):
        load_opts = load_opts or {}
        file_types = self.get_file_types(load_opts)
        # * Get paths
        paths = self.get_database_paths(database)
        res = deepcopy(self.DATA_STRUCTURE)

        for key in file_types:
            path = paths["save_dests"][db_type][key]
            if not path.exists():
                raise ValueError(
                    "Database file {} not found. Please run check_datasets() before".format(
                        str(path)
                    )
                )
            tmp = self.load_file(path)
            callback = load_opts.get("onload_callbacks", {}).get(key, None)
            if callback:
                tmp = callback(tmp)
            res[key] = tmp
        return res

    def get_database_options(self, name):
        for db in self.opts["databases"]:
            if db["name"] == name:
                return db
        return None

    def load_datasets(self, db_type, by_dataset=False, load_opts=None):
        res = {}
        # * Iterate over databases
        for database in self.opts["databases"]:
            # * Only load data if the give db_type is in the database definition
            if db_type in self.get_db_option("db_types", database, self.DB_TYPES):
                print(
                    "Loading {0} data for database: {1}".format(
                        db_type, database["name"]
                    )
                )
                res[database["name"]] = self.load_dataset(database, db_type, load_opts)

        if not by_dataset:
            res = self.merge_datasets(res)
        return res
