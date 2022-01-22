from copy import deepcopy
from pathlib import Path
import mouffet.utils.file as file_utils
from dlbd.applications.phenology.utils import score_models
from dlbd.utils import get_models_conf
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)
from dlbd.data.audio_data_handler import AudioDataHandler

from dlbd.evaluation import EVALUATORS
from dlbd.applications.phenology.phenology_evaluator import PhenologyEvaluator

EVALUATORS.register_evaluator("phenology", PhenologyEvaluator)


runs_path = Path("/home/vin/Doctorat/dev/dlbd/results/runs")


evaluation_config_path = (
    "/home/vin/Doctorat/dev/dlbd/config/examples/global_eval/evaluation_config.yaml"
)

evaluation_config = file_utils.load_config(evaluation_config_path)

res = []

for path in runs_path.iterdir():
    if path.is_dir():
        conf = deepcopy(evaluation_config)
        model_dir = path / "models"
        preds_dir = path / "predictions"
        conf["predictions_dir"] = preds_dir
        conf["id"] = path.name
        print(model_dir)
        model_opts = {"model_dir": model_dir}
        conf = get_models_conf(conf, model_opts, append=True)
        evaluator = SongDetectorEvaluationHandler(opts=conf, dh_class=AudioDataHandler)
        stats = evaluator.evaluate()
        res += stats

# TODO : Matches consolidation does not work

evaluator.opts["id"] = "global"
paths = evaluator.save_results(res)

score_models(paths["stats"], evaluation_config["evaluation_dir"])
