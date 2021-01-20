#%%

import os
import pathlib

import pandas as pd
from mouffet.options.model_options import ModelOptions
from mouffet.utils.model_handler import ModelHandler

from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation.song_detector_evaluator import SongDetectorEvaluator
from dlbd.models.CityNetTF2Dropout import CityNetTF2Dropout
from dlbd.training.spectrogram_sampler import SpectrogramSampler

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

src_root = pathlib.Path("/media/vin/Backup/PhD/Acoustics")
dest_root = pathlib.Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Tommy/"
)
file_list_path = dest_root / "recording_list_full.csv"

file_list = pd.read_csv(file_list_path)

print(file_list)

#%%

model_opts = ModelOptions(
    {
        "model_dir": "results/models",
        "name": "CityNetTF2_Dropout_resized",
        "class": CityNetTF2Dropout,
        "version": 1,
        "id": "{model--from_epoch}",
        "id_prefixes": {"model--from_epoch": "_fe-"},
        "model": {"from_epoch": 30},
    }
)

opts = {"n_mels": 32, "type": "mel", "n_fft": 2048, "hop_length": 1024}
model = ModelHandler.load_model(model_opts)

#%%

sampler = SpectrogramSampler(model.opts, balanced=False)
res = []
for row in file_list.itertuples():
    path = src_root / row.path
    spec, info = AudioDataHandler.load_raw_data(path, opts)
    tmp = SongDetectorEvaluator.classify_element(model, spec, info, sampler)
    res.append(tmp)

res_df = pd.concat(res)

res_df.to_feather("results/predictions/tommy_preds.feather")

# #%%

# import pandas as pd

# a = pd.read_feather("../results/predictions/tommy_preds.feather")
# print(a)
