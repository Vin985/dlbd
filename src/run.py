import logging

import pandas as pd
from mouffet.runs import RunHandler
from mouffet.training.training_handler import TrainingHandler

from dlbd.applications.phenology import PhenologyEvaluator
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation import EVALUATORS
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)

logging.basicConfig(level=logging.DEBUG)
pd.options.mode.chained_assignment = "raise"

EVALUATORS.register_evaluator(PhenologyEvaluator)

# import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"cd
# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)


run_handler = RunHandler(
    handler_classes={
        "training": TrainingHandler,
        "data": AudioDataHandler,
        "evaluation": SongDetectorEvaluationHandler,
    },
    default_args={"run_dir": "config/runs"},
)

run_handler.launch_runs()
