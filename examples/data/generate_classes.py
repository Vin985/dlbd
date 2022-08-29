#%%

from pathlib import Path

import pandas as pd

import mouffet.utils.file as file_utils

from dlbd.data.audio_data_handler import AudioDataHandler


opts_path = Path("../config/data_config.yaml")

opts = file_utils.load_config(opts_path)

dh = AudioDataHandler(opts)
# dh.check_datasets()
res = dh.get_summaries(load_opts={"file_types": ["infos", "tags_df"]})

#%%

agg_databases = []
agg_classes = []
classes_list = {}
all_classes = []
for db, summary in res.items():
    classes_list[db] = []
    for db_type, values in summary.items():
        classes = values.pop("classes_summary")
        classes["database"] = db
        classes["type"] = db_type
        agg_classes.append(classes)

        values["database"] = db
        values["type"] = db_type
        agg_databases.append(values)
        classes_list[db] += values["classes_list"]
        all_classes += values["classes_list"]

    classes_list[db] = list(set(classes_list[db]))

all_classes = list(set(all_classes))
all_classes.sort()

print(all_classes)

pd.DataFrame({"class_type": ["biotic"] * len(all_classes), "tag": all_classes}).to_csv(
    "generated_classes.csv", index=False
)


classes_df = pd.concat(agg_classes)
databases_df = pd.DataFrame(agg_databases)
