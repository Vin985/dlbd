#%%

from pathlib import Path

import pandas as pd

import mouffet.utils.file as file_utils

from dlbd.data.audio_data_handler import AudioDataHandler


opts_path = Path("examples/training_data_config.yaml")

opts = file_utils.load_config(opts_path)

dh = AudioDataHandler(opts)
dh.check_datasets()
res = dh.get_summaries()

#%%

# print(res)

agg_databases = []
agg_classes = []
classes_list = {}
all_classes = []
for key, val in res.items():
    classes_list[key] = []
    for key2, val2 in val.items():
        classes = val2.pop("classes_summary")
        classes["database"] = key
        classes["type"] = key2
        agg_classes.append(classes)

        val2["database"] = key
        val2["type"] = key2
        agg_databases.append(val2)
        classes_list[key] += val2["classes_list"]
        all_classes += val2["classes_list"]

    classes_list[key] = list(set(classes_list[key]))

all_classes = list(set(all_classes))
all_classes.sort()

print(all_classes)

pd.DataFrame({"class_type": ["biotic"] * len(all_classes), "tag": all_classes}).to_csv(
    "generated_classes.csv", index=False
)


classes_df = pd.concat(agg_classes)
databases_df = pd.DataFrame(agg_databases)


# print(classes_df)
# print(databases_df)


# print(classes_df.tag.unique())
