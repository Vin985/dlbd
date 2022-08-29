#%%

from pathlib import Path

import mouffet.utils.file as file_utils
import pandas as pd
from dlbd.data.audio_data_handler import AudioDataHandler

summaries_root = Path("/home/vin/Doctorat/dev/dlbd/results/summaries")

opts_path = Path("/home/vin/Doctorat/dev/dlbd/examples/config/data_config.yaml")
# opts_path = Path("../data_config.yaml")

opts = file_utils.load_config(opts_path)
opts["tags"]["filter_classes"] = False

if opts["tags"].get("filter_classes") == False:
    class_type = "no_filter"
else:
    class_type = opts["class_type"]

dh = AudioDataHandler(opts)
# dh.check_datasets(databases=[db])

#%%

res = dh.get_summaries(load_opts={"file_types": ["metadata", "tags_df"]})


#%%
all_tags = []
datasets = []

for db, summary in res.items():
    for db_type, values in summary.items():
        classes_summary = values.pop("classes_summary")
        file_path = file_utils.ensure_path_exists(
            summaries_root
            / ("_".join(["classes_summary", db, db_type, class_type]) + ".csv"),
            is_file=True,
        )
        classes_summary.to_csv(file_path)
        values.pop("raw_df")
        values["database"] = db
        values["type"] = db_type
        datasets.append(pd.DataFrame([values]))

res = pd.concat(datasets)
res_path = file_utils.ensure_path_exists(
    summaries_root / ("datasets_summary_" + class_type + ".csv"),
    is_file=True,
)
res.to_csv(res_path)
#%%


# all_df = pd.concat(all_tags)
# summary = dh.compute_summary(all_df)

# summary.to_csv(
#     file_utils.ensure_path_exists(
#         summaries_root / ("tag_summary_" + db + "_all.csv"), is_file=True
#     )
# )
