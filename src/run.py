import logging

import pandas as pd
from mouffet.runs import RunArgumentParser, launch_runs
from mouffet.training.training_handler import TrainingHandler

from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)

logging.basicConfig(level=logging.DEBUG)
pd.options.mode.chained_assignment = "raise"
# import tensorflow as tf


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"cd
# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)

parser = RunArgumentParser()
args = parser.parse_args()

launch_runs(
    args,
    handler_classes={
        "training": TrainingHandler,
        "data": AudioDataHandler,
        "evaluation": SongDetectorEvaluationHandler,
    },
)
