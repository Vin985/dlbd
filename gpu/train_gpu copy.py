import os

import tensorflow as tf
from mouffet.training.trainer import Trainer

from dlbd.data.audio_data_handler import AudioDataHandler
from dlbd.models.CityNetTF2Dropout import CityNetTF2Dropout

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

trainer = Trainer(
    # opts_path="src/CityNetTF2_Dropout_training_config.yaml",
    opts_path="src/CityNetTF2_Dropout_training_config_gpu.yaml",
    model_class=CityNetTF2Dropout,
    dh_class=AudioDataHandler,
)
# trainer.train_model()
trainer.train()

# stream = open("src/training_config.yaml", "r")
# opts = yaml.load(stream, Loader=yaml.Loader)

# data_opts_path = opts.get("data_config", "")
# if not data_opts_path:
#     raise Exception("A path to the data config file must be provided")
# dh = DataHandler(opts, split_funcs={"arctic": arctic_split})


# # # dh.create_datasets()
# # dh.check_datasets(split_funcs={"arctic": arctic_split})

# # train = dh.load_data("training")
# # validate = dh.load_data("validation")

# model = CityNetTF2(opts)
# # model = CityNetRegularized(opts)
# trainer = Trainer(opts, dh, model)
# trainer.train_model()

# for database in dh.opts["data"]["databases"]:
# print(dh.get_database_paths2(database, "train"))
# dh.create_dataset(database)
# paths = dh.get_database_paths(database)
# print(paths)
# arctic_split(paths, 0.2)

