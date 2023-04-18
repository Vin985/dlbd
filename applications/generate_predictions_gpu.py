#%%

import traceback
from pathlib import Path
import pandas as pd

import tensorflow as tf
from dlbd.evaluation import predictions
from dlbd.models import DLBD
from mouffet import common_utils
from mouffet.options.model_options import ModelOptions
from mouffet.utils.file import ensure_path_exists
from mouffet.utils.model_handler import ModelHandler

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


root_dir = Path("/mnt/PhD/Acoustics")
dest_root = Path("/mnt/PhD/Analysis/predictions_v2")


model_opts = ModelOptions(
    {
        "model_dir": "../resources/models/",
        "name": "DLBD_v2",
        "class": DLBD,
        "batch_size": 64,
        "spectrogram_overlap": 0.75,
        "inference": True,
        "random_start": False,
        "ignore_parent_path": True,
        "input_height": 32,
    }
)

overwrite = True


model = ModelHandler.load_model(model_opts)


spec_opts = {"n_fft": 512, "n_mels": 32, "sample_rate": "original", "to_db": False}

infos_res = []

years = [x for x in root_dir.iterdir() if x.is_dir()]
for year in years:
    sites = [x for x in year.iterdir() if x.is_dir()]
    for site in sites:
        plots = [x for x in site.iterdir() if x.is_dir()]
        for plot in plots:
            try:
                dest_path = (
                    dest_root
                    / f"{year.name}_{plot.name}_predictions_overlap-{model_opts['spectrogram_overlap']}.feather"
                )
                if not dest_path.exists() or overwrite:
                    wav_list = list(plot.glob("*.WAV"))
                    preds, infos = predictions.classify_elements(
                        wav_list, model, spec_opts
                    )
                    preds.reset_index().to_feather(
                        ensure_path_exists(dest_path, is_file=True)
                    )
                    infos["year"] = year.name
                    infos["site"] = site.name
                    infos["plot"] = plot.name
                    infos_res.append(infos)
                    tmp_stats = pd.DataFrame([infos])
                    tmp_stats.to_csv(
                        dest_root / f"{year.name}_{plot.name}_stats.csv", index=False
                    )
            except Exception:
                common_utils.print_error(traceback.format_exc())

classification_stats = pd.DataFrame(infos_res)
classification_stats.to_csv(dest_root / "global_stats.csv", index=False)

# %%
