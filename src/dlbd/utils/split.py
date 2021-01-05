from mouffet.utils.split import split_folder


def arctic_split(paths, data_handler=None, database=None):
    audio_path = paths["audio"]["training"]
    if not audio_path.exists():
        raise ValueError(
            "'audio_dir' option must be provided to split into test, training and"
            + "validation subsets"
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

