from pathlib import Path

import mouffet.utils.file as file_utils

from dlbd.data.audio_data_handler import AudioDataHandler

opts_path = Path("src/training_data_config.yaml")

opts = file_utils.load_config(opts_path)

dh = AudioDataHandler(opts)
dh.check_datasets(databases=["nips4b"])
