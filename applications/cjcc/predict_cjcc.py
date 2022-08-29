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
from dlbd.models.CityNetTF2Dropout2 import CityNetTF2Dropout
from dlbd.training.spectrogram_sampler import SpectrogramSampler

#%%
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)


plot = "BARW_0"
src_raw_root = (
    pathlib.Path(
        "/media/vin/Seagate Backup Plus Drive/Backup Acoustics Data/2018/Barrow/"
    )
    / plot
)
src_root = pathlib.Path("/mnt/win/UMoncton/Doctorat/dev/dlbd/results/predictions/CJCC")
dest_root = pathlib.Path("/mnt/win/UMoncton/Doctorat/dev/dlbd/results/predictions/CJCC")

file_list = src_raw_root.glob("*.WAV")

opts = {"n_mels": 64, "type": "mel", "n_fft": 2048, "hop_length": 512}


#%%
final_path = dest_root / (plot + ".pkl")
if not final_path.exists():
    print(final_path)
    tmp = {"specs": [], "infos": []}
    for file_path in file_list:
        print("Loading file: ", file_path)
        spec, info = AudioDataHandler.load_raw_data(file_path, opts)
        tmp["specs"].append(spec)
        tmp["infos"].append(info)

    with open(ensure_path_exists(final_path, is_file=True), "wb") as f:
        pickle.dump(tmp, f, -1)
        print("Saved file: ", final_path)


#%%

model_opts = ModelOptions(
    {
        "model_dir": "/mnt/win/UMoncton/Doctorat/dev/dlbd/results/models",
        "name": "CityNetTF2_Dropout_resized_mel64_hop_512",
        "class": CityNetTF2Dropout,
        "version": 19,
        "id": "{model--from_epoch}",
        "id_prefixes": {"model--from_epoch": "_fe-"},
        "model": {"from_epoch": 10},
    }
)

model = ModelHandler.load_model(model_opts)

#%%


sampler = SpectrogramSampler(model.opts, balanced=False)
res = []
i = 0
tic = time()
plot_tmp = []
pkl_path = src_root / (plot + ".pkl")
data = pickle.load(open(pkl_path, "rb"))
save_every = 100
for i, spec in enumerate(data["specs"]):
    tmp = SongDetectorEvaluator.classify_element(model, spec, data["infos"][i], sampler)
    plot_tmp.append(tmp)
    iters = len(plot_tmp)
    if iters % save_every == 0:
        tmp_data = plot_tmp[iters - save_every : iters]
        tmp_save = pd.concat(tmp_data).reset_index()
        tmp_save.to_feather(
            str(
                ensure_path_exists(
                    dest_root
                    / "intermediate"
                    / (
                        plot
                        + "_"
                        + str(iters - save_every)
                        + "-"
                        + str(iters)
                        + ".feather"
                    ),
                    is_file=True,
                )
            )
        )
tmp_df = pd.concat(plot_tmp).reset_index()
tmp_df.to_feather(
    str(ensure_path_exists(dest_root / (plot + ".feather"), is_file=True))
)
print("Classified {0} in {1}".format(plot, time() - tic))


# res_df = pd.concat(res).reset_index()
# res_df.to_feather(
#     str(ensure_path_exists(dest_root / "predictions_all.feather", is_file=True))
# )

# #%%

# import pandas as pd

# a = pd.read_feather("../results/predictions/tommy_preds.feather")
# print(a)
