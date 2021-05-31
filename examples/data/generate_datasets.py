from pathlib import Path

import mouffet.utils.file as file_utils

from dlbd.data.audio_data_handler import AudioDataHandler

import os

print(os.getcwd())

opts_path = Path("examples/training_data_config.yaml")

opts = file_utils.load_config(opts_path)

dh = AudioDataHandler(opts)
dh.check_datasets(databases=["nips4b"])
