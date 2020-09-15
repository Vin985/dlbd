import yaml
from pathlib import Path


from dlbd.data.data_handler import DataHandler
from dlbd.utils.split import arctic_split

stream = open("CONFIG.yaml", "r")
opts = yaml.load(stream, Loader=yaml.Loader)

dh = DataHandler(opts)


# dh.create_datasets()
dh.check_datasets()

# for database in dh.opts["data"]["databases"]:
# print(dh.get_database_paths2(database, "train"))
# dh.create_dataset(database)
# paths = dh.get_database_paths(database)
# print(paths)
# arctic_split(paths, 0.2)

