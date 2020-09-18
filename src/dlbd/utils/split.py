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


def split_list(data, split_props):
    splits = []
    tmp_list = data.copy()
    random.shuffle(tmp_list)
    for proportion in split_props:
        split_length = int(len(tmp_list) * proportion)
        splits.append(tmp_list[0:split_length])
        tmp_list = tmp_list[split_length:]
    splits.append(tmp_list)
    return splits


def split_folder(path, split_props):
    splits = [[] for i in range(len(split_props) + 1)]
    dirs, files = list_folder(path, [".wav"])
    if files:
        tmp_splits = split_list(files, split_props)
        for i, split in enumerate(tmp_splits):
            splits[i] += split
    if dirs:
        for dir_path in dirs:
            tmp_splits = split_folder(dir_path, split_props)
            for i, split in enumerate(tmp_splits):
                splits[i] += split
    return splits


def arctic_split(paths, data_handler=None, database=None):
    audio_path = paths["audio"]["training"]
    if not audio_path.exists():
        raise ValueError(
            "'audio_dir' option must be provided to split into test, training and validation subsets"
        )
    split = data_handler.get_db_option("split", database, None)
    if not split:
        raise ValueError("Split option must be provided for arctic split function")
    split_props = []
    # * Make test split optional
    test_split = split.get("test", 0)
    if test_split:
        split_props.append(test_split)
    val_split = split.get("validation", 0.2)
    split_props.append(val_split)
    splits = split_folder(audio_path, split_props)
    res = {}
    idx = 0
    if test_split:
        res["test"] = splits[idx]
        idx += 1
    res["validation"] = splits[idx]
    res["training"] = splits[idx + 1]

    print([(k + " " + str(len(v))) for k, v in res.items()])
    return res

    # Save file_list
    # data_handler.save_file_list("training", training, paths)
    # data_handler.save_file_list("validation", validation, paths)
    # return {"training": training, "validation": validation}
