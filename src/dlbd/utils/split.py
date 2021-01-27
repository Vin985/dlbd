import pandas as pd
from mouffet.utils.file import list_folder


def citynet_split(path, split_props, extensions):
    splits = [[] for i in range(len(split_props) + 1)]
    _, files = list_folder(path, extensions)
    file_names = [path.stem for path in files]
    groups = [name.split("_")[0] for name in file_names]
    df = pd.DataFrame({"paths": files, "groups": groups})
    tmp_df = df.copy()
    i = 0
    for split in split_props:
        split_df = tmp_df.groupby("groups").sample(frac=split)
        splits[i] = split_df.paths.to_list()
        tmp_df = tmp_df[~tmp_df.isin(split_df)].dropna()
        i += 1
    splits[i] = tmp_df.paths.to_list()
    return None
