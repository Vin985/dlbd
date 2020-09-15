import random

from .file import list_folder


def random_split(split, paths):
    audio_path = paths["training_audio_dir"]
    if not audio_path.exists():
        raise ValueError(
            "'audio_dir' option must be provided to split into training and validation subsets"
        )
    files = [str(p) for p in audio_path.rglob("*") if p.suffix.lower() == ".wav"]
    n_files = len(files)
    n_validation = int(split * n_files)
    validation_idx = random.sample(range(0, n_files), n_validation)
    training, validation = [], []
    for i in range(0, n_files):
        if i in validation_idx:
            validation.append(files[i])
        else:
            training.append(files[i])

    # Save file_list
    return {"training": training, "validation": validation}


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
    dirs, files = list_folder(path, [".wav"])
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
