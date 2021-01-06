import os
from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation.song_detector_evaluator import SongDetectorEvaluator


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


evaluator = SongDetectorEvaluator(
    opts_path="src/evaluation_config.yaml", dh_class=AudioDataHandler
)

res = evaluator.evaluate()
