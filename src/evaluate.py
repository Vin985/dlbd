import os
from dlbd.audio.audio_data_handler import AudioDataHandler
from dlbd.audio.song_detector_evaluator import SongDetectorEvaluator


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# stream = open("src/evaluation_config.yaml", "r")
# opts = yaml.load(stream, Loader=yaml.Loader)

# print(opts)

# for model in opts["models"]:
#     versions = model.get("versions", 1)
#     print(type(versions))

evaluator = SongDetectorEvaluator(
    opts_path="src/evaluation_config2.yaml", dh_class=AudioDataHandler
)

res = evaluator.evaluate()

# res.to_csv("results/evaluation/stats3.csv")
