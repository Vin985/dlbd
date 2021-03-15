#%%

import pandas as pd

std_events = pd.read_feather(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Tommy/std_events_95_15_300.feather"
)
std_events_agg = pd.read_feather(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Tommy/std_events_95_15_300_aggregate.feather"
)


#%%

tags_arctic = pd.read_feather(
    "/mnt/win/UMoncton/Doctorat/data/dl_training/datasets/arctic/original_mel_64_2048_512/training_tags_df_biotic.feather"
)

