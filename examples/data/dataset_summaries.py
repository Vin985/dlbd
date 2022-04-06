#%%

from pathlib import Path
import mouffet.utils.file as file_utils

from dlbd.data.audio_data_handler import AudioDataHandler

summaries_root = Path("/home/vin/Doctorat/dev/dlbd/results/summaries")

opts_path = Path("examples/data_config.yaml")
# opts_path = Path("../data_config.yaml")

opts = file_utils.load_config(opts_path)

db = "full_summer1"

dh = AudioDataHandler(opts)
# dh.check_datasets(databases=[db])
res = dh.get_db_tags_summary(db)

#%%

all_tags = []

for key, value in res.items():
    file_path = file_utils.ensure_path_exists(
        summaries_root / ("tag_summary_" + db + "_" + key + ".csv"), is_file=True
    )
    value["summary"].to_csv(file_path)

# all_df = pd.concat(all_tags)
# summary = dh.compute_summary(all_df)

# summary.to_csv(
#     file_utils.ensure_path_exists(
#         summaries_root / ("tag_summary_" + db + "_all.csv"), is_file=True
#     )
# )
