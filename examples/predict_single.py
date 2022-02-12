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

from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)

src_root_dir = Path()

dest_root = pathlib.Path("/mnt/win/UMoncton/Doctorat/dev/dlbd/results/predict")


model_opts = ModelOptions(
    {
        "model_dir": "/home/vin/Desktop/results/summary/",
        # "name": "DLBDL_wr3_fil128_d1-64_d2-32_pps200_bs256",
        "name": "DLBDL",
        "id": "_wr3_fil64_d1-512_d2-256_pps100_bs256",
        "class": DLBD,
        "batch_size": 256,
        "spectrogram_overlap": 0.95,
        "inference": True,
        "random_start": False,
        # "A": 0.05,
        # "B": 10,
    }
)

model = ModelHandler.load_model(model_opts, ignore_parent_path=True)


print(model.opts.opts)

spec_opts = {"n_fft": 512, "n_mels": 32, "sample_rate": "original", "to_db": False}

file_path = Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT12_EC20/20210706_013000.WAV"
)
# file_path = Path(
#     "/mnt/win/UMoncton/Doctorat/data/dl_training/raw/full_summer_subset1/IGLO_B_2018-06-12_232955_247.wav"
# )

preds, info = predictions.classify_elements([file_path], model, spec_opts)

#%%
preds.plot("time", "activity")

#%%

preds_eval = pd.read_feather(
    "/mnt/win/UMoncton/Doctorat/dev/dlbd/examples/phenology/results/predictions/original_mel32_512_None_None/full_summer1_DLBDL_wr3_fil64_d1-512_d2-256_pps100_bs256_v1.feather"
)


# import pickle

# specs = pickle.load(
#     open(
#         "/mnt/win/UMoncton/Doctorat/data/dl_training/datasets/full_summer1/original_mel32_512_None_None/test_spectrograms.pkl",
#         "rb",
#     )
# )

# #%%
# infos = pickle.load(
#     open(
#         "/mnt/win/UMoncton/Doctorat/data/dl_training/datasets/full_summer1/original_mel32_512_None_None/test_infos.pkl",
#         "rb",
#     )
# )

# #%%


# for i, info in enumerate(infos):
#     if str(info["file_path"]).endswith("955_247.wav"):
#         spec2 = specs[i]
#         break


# print(spec2)

# #%%
# import numpy as np

# print(np.all(spec == spec2))


# {
#     "A": 0.001,
#     "B": 10.0,
#     "batch_size": 256,
#     "channels": 4,
#     "clean_empty_models": True,
#     "conv_filter_width": 4,
#     "databases_options": {
#         "spectrogram": {
#             "n_fft": 512,
#             "n_mels": 32,
#             "sample_rate": "original",
#             "to_db": False,
#         }
#     },
#     "dilation_rate": [[1, 2], 1],
#     "do_augmentation": 1,
#     "do_batch_norm": 1,
#     "dropout": 0.5,
#     "early_stopping": {
#         "min_delta": 0.002,
#         "patience": 8,
#         "restore_best_weights": False,
#     },
#     "ensemble_members": 1,
#     "epoch_save_step": 3,
#     "freq_mask": True,
#     "input_height": 32,
#     "input_width": 100,
#     "learn_log": 0,
#     "learning_rate": 0.01,
#     "n_epochs": 50,
#     "name": "DLBDL_wr3_fil64_d1-512_d2-256_pps100_bs256",
#     "num_dense_units": 512,
#     "num_dense_units2": 256,
#     "num_filters": 64,
#     "pixels_per_sec": 100,
#     "random_start": True,
#     "resize_spectrogram": True,
#     "skip_trained": True,
#     "spectrogram_overlap": 0.95,
#     "time_mask": True,
#     "training": True,
#     "training_balanced": True,
#     "wiggle_room": 10,
#     "wriggle_room": 3,
# }

# {
#     "n_epochs": 50,
#     "epoch_save_step": 3,
#     "learning_rate": 0.01,
#     "learn_log": 0,
#     "do_augmentation": 1,
#     "A": 0.001,
#     "B": 10.0,
#     "ensemble_members": 1,
#     "training_balanced": True,
#     "resize_spectrogram": True,
#     "pixels_per_sec": 100,
#     "spectrogram_overlap": 0.75,
#     "batch_size": 256,
#     "do_batch_norm": 1,
#     "channels": 4,
#     "num_filters": 64,
#     "num_dense_units": 512,
#     "conv_filter_width": 4,
#     "wiggle_room": 10,
#     "dropout": 0.5,
#     "databases_options": {
#         "spectrogram": {
#             "n_fft": 512,
#             "n_mels": 32,
#             "sample_rate": "original",
#             "to_db": False,
#         },
#         "class_type": "biotic",
#     },
#     "class": "dlbd.models.dlbd.DLBD",
#     "dilation_rate": [[1, 2], 1],
#     "early_stopping": {
#         "min_delta": 0.002,
#         "patience": 8,
#         "restore_best_weights": False,
#     },
#     "freq_mask": True,
#     "input_height": 32,
#     "input_width": 100,
#     "name": "DLBDL",
#     "num_dense_units2": 256,
#     "parent_path": "config/gpu/runs/training_parent.yaml",
#     "random_start": True,
#     "skip_trained": True,
#     "time_mask": True,
#     "training": True,
#     "wriggle_room": 3,
#     "reclassify": False,
#     "models_options": {"spectrogram_overlap": 0.95, "reclassify": False},
#     "inference": True,
# }


# {
#     "do_augmentation": False,
#     "learn_log": 0,
#     "hww_spec": 50,
#     "hww_gt": 50,
#     "batch_size": 256,
#     "overlap": 0.95,
#     "random_start": True,
#     "gt_prop": 0,
#     "randomise": False,
#     "balanced": False,
# }
# {
#     "do_augmentation": False,
#     "learn_log": 0,
#     "hww_spec": 50,
#     "hww_gt": 50,
#     "batch_size": 256,
#     "overlap": 0.75,
#     "random_start": True,
#     "gt_prop": 0,
#     "randomise": False,
#     "balanced": False,
# }
