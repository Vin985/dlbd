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

EVALUATORS.register_evaluator("phenology", PhenologyEvaluator)


#%%

models_dir = Path("/home/vin/Desktop/results/candidates_models")

evaluation_config_path = (
    "/home/vin/Doctorat/dev/dlbd/config/examples/phenology/evaluation_config.yaml"
)

evaluation_config = file_utils.load_config(evaluation_config_path)


model_opts = {"model_dir": models_dir}

evaluation_config = get_models_conf(evaluation_config, model_opts)


evaluator = SongDetectorEvaluationHandler(
    opts=evaluation_config, dh_class=AudioDataHandler
)

stats = evaluator.evaluate()
