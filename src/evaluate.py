import os
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)

import pandas as pd
import tensorflow as tf

pd.options.mode.chained_assignment = "raise"

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

evaluator = SongDetectorEvaluationHandler(
    opts_path="config/runs/run1/evaluation_config.yaml", dh_class=AudioDataHandler
)

res = evaluator.evaluate()
