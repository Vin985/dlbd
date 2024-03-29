#%%
import os
from pathlib import Path

import tensorflow as tf
from mouffet import config_utils, file_utils

from dlbd.applications.phenology.phenology_evaluator import PhenologyEvaluator
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation import EVALUATORS
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)

EVALUATORS.register_evaluator(PhenologyEvaluator)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

#%%

models_dir = "../resources/models"

evaluation_config_path = "config/phenology/evaluation_config_bench.yaml"

evaluation_config = file_utils.load_config(evaluation_config_path)


# evaluation_config["models_list_dir"] = models_dir

evaluation_config = config_utils.get_models_conf(
    evaluation_config,
    # updates={
    #     "model_dir": "resources/models",
    #     # "ignore_parent_path": True,
    #     "spectrogram_overlap": 0.5,
    #     # "reclassify": True,
    # },
)


evaluator = SongDetectorEvaluationHandler(
    opts=evaluation_config, dh_class=AudioDataHandler
)

stats = evaluator.evaluate()
