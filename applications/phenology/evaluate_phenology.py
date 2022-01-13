#%%

from pathlib import Path

import mouffet.utils.file as file_utils
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation import EVALUATORS
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)

from phenology_evaluator import PhenologyEvaluator
from utils import check_models

EVALUATORS.register_evaluator("phenology", PhenologyEvaluator)


#%%

models_dir = Path("/home/vin/Desktop/results/candidates_models")

evaluation_config_path = (
    "/home/vin/Doctorat/dev/dlbd/applications/phenology/evaluation_config.yaml"
)

evaluation_config = file_utils.load_config(evaluation_config_path)


model_opts = {"model_dir": models_dir}

evaluation_config = check_models(evaluation_config, model_opts)


evaluator = SongDetectorEvaluationHandler(
    opts=evaluation_config, dh_class=AudioDataHandler
)

stats = evaluator.evaluate()
