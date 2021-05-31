from pathlib import Path

import pandas as pd

import mouffet.utils.file as file_utils

from dlbd.data.audio_data_handler import AudioDataHandler


opts_path = Path("examples/training_data_config.yaml")

opts = file_utils.load_config(opts_path)

dh = AudioDataHandler(opts)
dh.check_datasets()
res = dh.get_summaries()

# print(res)

agg_databases = []
agg_classes = []
for key, val in res.items():
    for key2, val2 in val.items():
        classes = val2.pop("classes")
        classes["database"] = key
        classes["type"] = key2
        agg_classes.append(classes)

        val2["database"] = key
        val2["type"] = key2
        agg_databases.append(val2)

classes_df = pd.concat(agg_classes)
databases_df = pd.DataFrame(agg_databases)

print(classes_df)
print(databases_df)

