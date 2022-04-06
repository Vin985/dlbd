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
from dlbd.data import audio_utils

from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.training.spectrogram_sampler import SpectrogramSampler
from dlbd.models import DLBD
from dlbd.evaluation import predictions


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)


plots = [
    {
        "src_path": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT1_EC29",
        "name": "LYT1_EC29",
    },
    {
        "src_path": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT12_EC20",
        "name": "LYT12_EC20",
    },
    {
        "src_path": "/media/vin/BigMama/Sylvain/AL57",
        "name": "2021_BARW_0_AL57",
    },
    {
        "src_path": "/media/vin/BigMama/Sylvain/AL58",
        "name": "2021_BARW_8_AL58",
    },
    {
        "src_path": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT2_EC09",
        "name": "LYT2_EC09",
    },
    {
        "src_path": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT8_EC06",
        "name": "LYT8_EC06",
    },
    {
        "src_path": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT9_AL47",
        "name": "LYT9_AL47",
    },
]

dest_root = pathlib.Path("/mnt/win/UMoncton/Doctorat/dev/dlbd/results/predict")


model_opts = ModelOptions(
    {
        "model_dir": "/home/vin/Desktop/results/summary/",
        # "name": "DLBDL_wr3_fil128_d1-64_d2-32_pps200_bs256",
        "name": "DLBDL",
        "id": "_wr3_fil64_d1-512_d2-256_pps100_bs256",
        "class": DLBD,
        "inference": True,
        "random_start": False,
    }
)

model = ModelHandler.load_model(model_opts)
overwrite = False

spec_opts = {"n_fft": 512, "n_mels": 32, "sample_rate": "original", "to_db": False}

infos_res = []
for plot in plots:
    dest_path = pathlib.Path(dest_root) / plot["name"] / "predictions.feather"
    if not dest_path.exists() or overwrite:
        preds, infos = predictions.classify_elements(
            list(pathlib.Path(plot["src_path"]).glob("*.WAV")), model, spec_opts
        )
        preds.reset_index().to_feather(ensure_path_exists(dest_path, is_file=True))
        infos_res.append(infos)

# pd.DataFrame(infos_res).to_csv(
#     ensure_path_exists(dest_root / "infos.csv", is_file=True)
# )


# for audio_file in src_audio_root.glob("*.WAV"):
#     print(audio_file)
#     spec, info = audio_utils.load_audio_data(audio_file, spec_opts)


# #%%
# for plot_id in file_list.plot_id.unique():
#     final_path = dest_root / (plot_id + ".pkl")
#     if not final_path.exists():
#         print(final_path)
#         tmp = {"specs": [], "infos": []}
#         tmp_df = file_list.loc[file_list.plot_id == plot_id]
#         for row in tmp_df.itertuples():
#             path = src_raw_root / row.path
#             print("Loading file: ", path)
#             spec, info = AudioDataHandler.load_raw_data(path, opts)
#             tmp["specs"].append(spec)
#             tmp["infos"].append(info)

#         with open(ensure_path_exists(final_path, is_file=True), "wb") as f:
#             pickle.dump(tmp, f, -1)
#             print("Saved file: ", final_path)


# #%%

# model_opts = ModelOptions(
#     {
#         "model_dir": "results/models",
#         "name": "CityNetTF2_Dropout_resized",
#         "class": CityNetTF2Dropout,
#         "version": 1,
#         "id": "{model--from_epoch}",
#         "id_prefixes": {"model--from_epoch": "_fe-"},
#         "model": {"from_epoch": 30},
#     }
# )

# model = ModelHandler.load_model(model_opts)

# #%%


# sampler = SpectrogramSampler(model.opts, balanced=False)
# res = []
# i = 0
# for plot_id in file_list.plot_id.unique():
#     tic = time()
#     plot_tmp = []
#     pkl_path = src_root / (plot_id + ".pkl")
#     data = pickle.load(open(pkl_path, "rb"))
#     for i, spec in enumerate(data["specs"]):
#         tmp = SongDetectorEvaluator.classify_element(
#             model, spec, data["infos"][i], sampler
#         )
#         plot_tmp.append(tmp)
#         break
#     tmp_df = pd.concat(plot_tmp).reset_index()
#     res.append(tmp_df)
#     tmp_df.to_feather(
#         str(
#             ensure_path_exists(
#                 dest_root / "intermediate" / (plot_id + ".feather"), is_file=True
#             )
#         )
#     )
#     print("Classified {0} in {1}".format(plot_id, time() - tic))


# res_df = pd.concat(res).reset_index()
# res_df.to_feather(
#     str(ensure_path_exists(dest_root / "predictions_all.feather", is_file=True))
# )

# #%%

# import pandas as pd

# a = pd.read_feather("../results/predictions/tommy_preds.feather")
# print(a)
