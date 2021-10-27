import os
import logging

# import tensorflow as tf
from mouffet.training.training_handler import TrainingHandler

from dlbd.data.audio_data_handler import AudioDataHandler

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)

logging.basicConfig(level=logging.DEBUG)

trainer = TrainingHandler(
    opts_path="config/runs/run1/training_config.yaml",
    dh_class=AudioDataHandler,
)
trainer.train()
