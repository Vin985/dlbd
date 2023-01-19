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

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


dest_root = pathlib.Path(
    "/mnt/win/UMoncton/Doctorat/dev/dlbd/examples/results/predict_multiple_bench_gpu"
)

batch_sizes = [64, 128, 256]
spectrogram_overlaps = [0, 0.3, 0.5, 0.7, 0.95]
infos_res = []

for bs in batch_sizes:
    for overlap in spectrogram_overlaps:

        model_opts = ModelOptions(
            {
                "model_dir": "/mnt/win/UMoncton/Doctorat/dev/dlbd/resources/models/",
                "name": "DLBD",
                "class": DLBD,
                "batch_size": bs,
                "inference": True,
                "random_start": False,
                "ignore_parent_path": True,
                "spectrogram_overlap": overlap,
                # "pixels_per_sec": 100,
                # "input_height": 32,
                # "wiggle_room":
            }
        )

        plots = [
            {
                "src_path": "/mnt/win/UMoncton/Doctorat/data/acoustic/field/2018/Plot1",
                "name": f"test_bench_overlap{overlap}_bs{bs}",
            },
        ]

        model = ModelHandler.load_model(model_opts)
        overwrite = True

        spec_opts = {
            "n_fft": 512,
            "n_mels": 32,
            "sample_rate": "original",
            "to_db": False,
        }

        for plot in plots:
            dest_path = (
                pathlib.Path(dest_root)
                / plot["name"]
                / f"predictions_overlap{overlap}_bs{bs}.feather"
            )
            if not dest_path.exists() or overwrite:
                preds, infos = predictions.classify_elements(
                    list(pathlib.Path(plot["src_path"]).glob("*.WAV")),
                    model,
                    spec_opts,
                )
                preds.reset_index().to_feather(
                    ensure_path_exists(dest_path, is_file=True)
                )
                infos["plot"] = plot["name"]
                infos_res.append(infos)

infos_df = pd.DataFrame(infos_res).reset_index(drop=True)
stats_file = dest_root / "predictions_stats.csv"
if stats_file.exists():
    stats_df = pd.read_csv(stats_file)
    infos_df = pd.concat([stats_df, infos_df]).drop_duplicates(["plot"], keep="last")
infos_df.reset_index(drop=True).to_csv(stats_file, index=False)
