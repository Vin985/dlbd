import os
from dlbd.audio.audio_data_handler import AudioDataHandler
from dlbd.audio.models.CityNetTF2Dropout import CityNetTF2Dropout

from dlbd.training.trainer import Trainer
from dlbd.utils.split import arctic_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

trainer = Trainer(
    # opts_path="src/CityNetTF2_Dropout_training_config.yaml",
    opts_path="CityNetTF2_Dropout_training_config_term.yaml",
    model=CityNetTF2Dropout(),
    dh_class=AudioDataHandler,
    split_funcs={"arctic": arctic_split},
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

