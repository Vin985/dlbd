#%%
import os

import pathlib
import pandas as pd

import tensorflow as tf

from mouffet.options.model_options import ModelOptions
from mouffet.utils.file import ensure_path_exists
from mouffet.utils.model_handler import ModelHandler

from dlbd.models import DLBD
from dlbd.evaluation import predictions

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)


plots = [
    # {
    #     "src_path": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT1_EC29",
    #     "name": "LYT1_EC29",
    # },
    # {
    #     "src_path": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT12_EC20",
    #     "name": "LYT12_EC20",
    # },
    # {
    #     "src_path": "/media/vin/BigMama/Sylvain/AL57",
    #     "name": "2021_BARW_0_AL57",
    # },
    # {
    #     "src_path": "/media/vin/BigMama/Sylvain/AL58",
    #     "name": "2021_BARW_8_AL58",
    # },
    # {
    #     "src_path": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT2_EC09",
    #     "name": "LYT2_EC09",
    # },
    # {
    #     "src_path": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT8_EC06",
    #     "name": "LYT8_EC06",
    # },
    # {
    #     "src_path": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Ny-Aalesund/2021/LYT9_AL47",
    #     "name": "LYT9_AL47",
    # },
    {
        # "src_path": "/media/vin/Backup/PhD/Acoustics/2018/Barrow/BARW_0",
        "src_path": "/mnt/win/UMoncton/Doctorat/data/acoustic/field/2018/Plot1",
        # "name": "BARW_0",
        "name": "test_bench",
    },
]

dest_root = pathlib.Path(
    # "/mnt/win/UMoncton/Doctorat/dev/dlbd/examples/results/predict_multiple"
    "/mnt/win/UMoncton/Doctorat/dev/dlbd/examples/results/predict_multiple_bench_cpu"
)


model_opts = ModelOptions(
    {
        "model_dir": "/mnt/win/UMoncton/Doctorat/dev/dlbd/resources/models/",
        "name": "DLBD",
        "class": DLBD,
        "batch_size": 64,
        "spectrogram_overlap": 0.95,
        "inference": True,
        "random_start": False,
        "ignore_parent_path": True,
        # "pixels_per_sec": 100,
        # "input_height": 32,
        # "wiggle_room":
    }
)

model = ModelHandler.load_model(model_opts)
overwrite = False

spec_opts = {"n_fft": 512, "n_mels": 32, "sample_rate": "original", "to_db": False}

infos_res = []
for plot in plots:
    dest_path = (
        pathlib.Path(dest_root)
        / plot["name"]
        / f"predictions_overlap{model_opts["spectrogram_overlap"]}.feather"
    )
    if not dest_path.exists() or overwrite:
        preds, infos = predictions.classify_elements(
            list(pathlib.Path(plot["src_path"]).glob("*.WAV")), model, spec_opts
        )
        preds.reset_index().to_feather(ensure_path_exists(dest_path, is_file=True))
        infos["plot"] = plot["name"]
        infos_res.append(infos)

infos_df = pd.DataFrame(infos_res)
infos_df.reset_index(drop=True).to_csv(dest_root / "predictions_stats.csv")
