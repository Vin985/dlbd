#%%

from pathlib import Path

import mouffet.utils.file as file_utils
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation import EVALUATORS
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)

from dlbd.applications.phenology.phenology_evaluator import PhenologyEvaluator
from dlbd.utils import get_models_conf

EVALUATORS.register_evaluator(PhenologyEvaluator)


#%%

models_dir = "resources/models"

evaluation_config_path = "config/phenology/evaluation_config.yaml"

evaluation_config = file_utils.load_config(evaluation_config_path)


# evaluation_config["models_list_dir"] = models_dir

evaluation_config = get_models_conf(
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
