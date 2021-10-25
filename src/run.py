import argparse
import logging

# import os
import sys
import time
from pathlib import Path

import mouffet.utils.file as file_utils
import pandas as pd
from mouffet.training.training_handler import TrainingHandler

from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.evaluation.song_detector_evaluation_handler import (
    SongDetectorEvaluationHandler,
)

# import tensorflow as tf


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)


logging.basicConfig(level=logging.DEBUG)
pd.options.mode.chained_assignment = "raise"


parser = argparse.ArgumentParser(
    description="Perform training and evaluation runs in one go"
)

parser.add_argument(
    "runs", metavar="run", type=str, nargs="+", help="the name of the run"
)

parser.add_argument(
    "-r",
    "--run_dir",
    default="config/runs",
    help="The root directory where runs can be found",
)

parser.add_argument(
    "-d",
    "--dest_dir",
    default="results/runs",
    help="The root directory where results will be saved",
)

parser.add_argument(
    "-t",
    "--training_config",
    default="training_config.yaml",
    help="The name of the training config files",
)

parser.add_argument(
    "-e",
    "--evaluation_config",
    default="evaluation_config.yaml",
    help="The name of the evaluation config files",
)

parser.add_argument(
    "-D",
    "--data_config",
    default="data_config.yaml",
    help="The name of the data config files",
)


parser.add_argument(
    "-l",
    "--log_dir",
    default="logs/runs",
    help="The root directory where logs will be saved",
)

args = parser.parse_args()


for run in args.runs:
    opts_path = Path(args.run_dir) / run
    dest_dir = Path(args.dest_dir) / run
    log_dir = Path(args.log_dir) / run
    run_opts = {}

    trainer = TrainingHandler(
        opts_path=opts_path / args.training_config,
        dh_class=AudioDataHandler,
    )

    for training_scenario in trainer.scenarios:
        if not "model_dir" in training_scenario:
            training_scenario["model_dir"] = dest_dir / "models"
        if not "data_config" in training_scenario:
            training_scenario["data_config"] = opts_path / args.data_config
        if not "log_dir" in training_scenario:
            training_scenario["log_dir"] = log_dir
        trainer.train_scenario(training_scenario)

    # evaluator = SongDetectorEvaluationHandler(
    #     opts_path=opts_path / args.evaluation_config, dh_class=AudioDataHandler
    # )
    print("toto")
    # trainer.train()
    # res = evaluator.evaluate()
