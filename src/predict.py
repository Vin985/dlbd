#%%

import os
import pathlib
import pickle
from time import time

import pandas as pd
import tensorflow as tf
from mouffet.options.model_options import ModelOptions
from mouffet.utils.file import ensure_path_exists
from mouffet.utils.model_handler import ModelHandler

from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation.song_detector_evaluator import SongDetectorEvaluator
from dlbd.models.CityNetTF2Dropout import CityNetTF2Dropout
from dlbd.training.spectrogram_sampler import SpectrogramSampler

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

src_root = pathlib.Path("/media/vin/Backup/PhD/Acoustics")
dest_root = pathlib.Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Tommy/"
)
file_list_path = dest_root / "recording_list_full.csv"

file_list = pd.read_csv(file_list_path)
file_list["plot_id"] = file_list.year.map(str) + "_" + file_list["plot"]

print(file_list)
opts = {"n_mels": 32, "type": "mel", "n_fft": 2048, "hop_length": 1024}

#%%


#%%
for plot_id in file_list.plot_id.unique():
    final_path = dest_root / (plot_id + ".pkl")
    if not final_path.exists():
        print(final_path)
        tmp = {"specs": [], "infos": []}
        tmp_df = file_list.loc[file_list.plot_id == plot_id]
        for row in tmp_df.itertuples():
            path = src_root / row.path
            print("Loading file: ", path)
            spec, info = AudioDataHandler.load_raw_data(path, opts)
            tmp["specs"].append(spec)
            tmp["infos"].append(info)

        with open(ensure_path_exists(final_path, is_file=True), "wb") as f:
            pickle.dump(tmp, f, -1)
            print("Saved file: ", final_path)


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

model = ModelHandler.load_model(model_opts)

#%%


sampler = SpectrogramSampler(model.opts, balanced=False)
res = []
i = 0
for plot_id in file_list.plot_id.unique():
    tic = time()
    plot_tmp = []
    pkl_path = dest_root / (plot_id + ".pkl")
    data = pickle.load(open(pkl_path, "rb"))
    for i, spec in enumerate(data["specs"]):
        tmp = SongDetectorEvaluator.classify_element(
            model, spec, data["infos"][i], sampler
        )
        plot_tmp.append(tmp)
    tmp_df = pd.concat(plot_tmp).reset_index()
    res.append(tmp_df)
    tmp_df.to_feather(str(dest_root) + "/intermediate/" + plot_id + ".feather")
    print("Classified {0} in {1}".format(plot_id, time() - tic))


res_df = pd.concat(res).reset_index()
res_df.to_feather(str(dest_root) + "predictions_all.feather")

# #%%

# import pandas as pd

# a = pd.read_feather("../results/predictions/tommy_preds.feather")
# print(a)
