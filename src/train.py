import os

# import tensorflow as tf
from mouffet.training.trainer import Trainer

from dlbd.data.audio_data_handler import AudioDataHandler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)

trainer = Trainer(
    opts_path="src/training_config_test.yaml",
    # opts_path="CityNetTF2_Dropout_training_config.yaml",
    # model_class=DLBDLite,
    dh_class=AudioDataHandler,
)
trainer.train()
