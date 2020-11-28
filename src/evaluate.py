import os
from dlbd.audio.audio_data_handler import AudioDataHandler
from dlbd.audio.audiodetector_evaluator import AudioDetectorEvaluator


from dlbd.evaluation.evaluator import Evaluator


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# stream = open("src/evaluation_config.yaml", "r")
# opts = yaml.load(stream, Loader=yaml.Loader)

# print(opts)

# for model in opts["models"]:
#     versions = model.get("versions", 1)
#     print(type(versions))

evaluator = AudioDetectorEvaluator(
    opts_path="src/evaluation_config.yaml", dh_class=AudioDataHandler
)

res = evaluator.evaluate()

# res.to_csv("results/evaluation/stats3.csv")
print(res)
