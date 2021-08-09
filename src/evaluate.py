import os
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)

import pandas as pd


pd.options.mode.chained_assignment = "raise"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


evaluator = SongDetectorEvaluationHandler(
    opts_path="src/evaluation_config_new.yaml", dh_class=AudioDataHandler
)

res = evaluator.evaluate()
