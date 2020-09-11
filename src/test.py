import random

import yaml
from pathlib import Path


from dlbd.data.data_handler import DataHandler

stream = open("CONFIG.yaml", "r")
opts = yaml.load(stream, Loader=yaml.Loader)

dh = DataHandler(opts)


def path_walk(path, extensions=None):
    dirs = []
    files = []
    extensions = extensions or []
    for item in Path(path).iterdir():
        if item.is_dir():
            dirs.append(item)
        else:
            if item.is_file() and item.suffix.lower() in extensions:
                files.append(item)
    return (dirs, files)


def split_files(files, proportion):
    split1, split2 = [], []
    n_files = len(files)
    n_split = int(proportion * n_files)
    split_idx = random.sample(range(0, n_files), n_split)
    for i in range(0, n_files):
        if i in split_idx:
            split2.append(files[i])
        else:
            split1.append(files[i])
    return (split1, split2)


def split_folder(path, split, data_handler, database):
    split1, split2 = [], []
    dirs, files = path_walk(path, [".wav"])
    if files:
        tmp_train, tmp_val = split_files(files, split)
        split1 += tmp_train
        split2 += tmp_val
    if dirs:
        for dir_path in dirs:
            tmp_train, tmp_val = split_folder(dir_path, split, data_handler, database)
            split1 += tmp_train
            split2 += tmp_val
    return (split1, split2)


def arctic_split(paths, split, data_handler=None, database=None):
    audio_path = paths["training_audio_dir"]
    if not audio_path.exists():
        raise ValueError(
            "'audio_dir' option must be provided to split into training and validation subsets"
        )
    training, validation = split_folder(audio_path, split, data_handler, database)

    print(training, validation)
    print(len(training), len(validation))
    # Save file_list
    # data_handler.save_file_list("training", training, paths)
    # data_handler.save_file_list("validation", validation, paths)
    # return {"training": training, "validation": validation}


# dh.create_datasets()


for database in dh.opts["data"]["databases"]:
    # print(dh.get_database_paths2(database, "train"))
    # dh.create_dataset(database)
    paths = dh.get_database_paths(database)
    arctic_split(paths, 0.2)

