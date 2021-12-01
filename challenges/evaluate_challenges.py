#%%

import ast
from pathlib import Path

import mouffet.utils.file as file_utils
import pandas as pd
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)
from mouffet.training.training_handler import TrainingHandler

wavs_dir = "wavs"
tags_file = "tags.csv"
warblr_dir = Path("/mnt/win/UMoncton/Doctorat/data/dl_training/raw/warblr/test")
models_dir = Path("/home/vin/Desktop/results/candidates_models")

evaluation_config_path = "challenges/evaluation_config.yaml"

evaluation_config = file_utils.load_config(evaluation_config_path)

models = evaluation_config.get("models", [])
if not models:
    models_stats_path = Path(models_dir / TrainingHandler.MODELS_STATS_FILE_NAME)
    models_stats = None
    if models_stats_path.exists():
        models_stats = pd.read_csv(models_stats_path).drop_duplicates(
            "opts", keep="last"
        )
    if models_stats is not None:
        models = [ast.literal_eval(row.opts) for row in models_stats.itertuples()]
        evaluation_config["models"] = models


evaluator = SongDetectorEvaluationHandler(
    opts=evaluation_config, dh_class=AudioDataHandler
)
evaluator.evaluate()
