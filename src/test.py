import os
import yaml


from dlbd.data.data_handler import DataHandler
from dlbd.training.trainer import Trainer
from dlbd.utils.split import arctic_split
from dlbd.models.CityNetTF2 import CityNetTF2
from dlbd.models.CityNetRegularized import CityNetRegularized


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

stream = open("src/CONFIG.yaml", "r")
opts = yaml.load(stream, Loader=yaml.Loader)

dh = DataHandler(opts, split_funcs={"arctic": arctic_split})


# # dh.create_datasets()
# dh.check_datasets(split_funcs={"arctic": arctic_split})

# train = dh.load_data("training")
# validate = dh.load_data("validation")

model = CityNetTF2(opts)
# model = CityNetRegularized(opts)
trainer = Trainer(opts, dh, model)
trainer.train_model()

# for database in dh.opts["data"]["databases"]:
# print(dh.get_database_paths2(database, "train"))
# dh.create_dataset(database)
# paths = dh.get_database_paths(database)
# print(paths)
# arctic_split(paths, 0.2)

