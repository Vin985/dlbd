from copy import deepcopy
from pathlib import Path
import pandas as pd
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
evaluation_config["save_use_date_subfolder"] = False
evaluation_config["save_use_time_prefix"] = False

res_dir = Path(evaluation_config["evaluation_dir"])
if not res_dir:
    raise ValueError(
        "You should specify a evaluation_dir folder in the evaluation_config file"
        + "for global evaluation"
    )

print(res_dir)

global_evaluator = SongDetectorEvaluationHandler(
    opts=evaluation_config, dh_class=AudioDataHandler
)
global_evaluator.opts["id"] = "global"
global_res_filename = "global_stats.csv"
global_res_path = res_dir / global_res_filename
overwrite = evaluation_config.get("overwrite_results", False)


res = []
if not global_res_path.exists() or overwrite:
    print("No global file saved or overwrite is selected, creating it")
    for path in runs_path.iterdir():
        if path.is_dir():
            save_path = res_dir / (path.name + "_stats.csv")
            if save_path.exists() and not overwrite:
                stats = []
                stats.append({"stats": pd.read_csv(save_path)})
            else:
                conf = deepcopy(evaluation_config)
                model_dir = path / "models"
                preds_dir = path / "predictions"
                conf["predictions_dir"] = preds_dir
                conf["id"] = path.name
                print(model_dir)
                model_opts = {"model_dir": model_dir}
                conf = get_models_conf(conf, model_opts, append=True)
                evaluator = SongDetectorEvaluationHandler(
                    opts=conf, dh_class=AudioDataHandler
                )
                stats = evaluator.evaluate()
            res += stats
    filenames = global_evaluator.save_results(res)
    global_res_filename = filenames["stats"]

# TODO : Matches consolidation does not work
score_models(global_res_filename, res_dir, evaluation_config)
