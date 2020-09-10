import yaml

from dlbd.data.data_handler import DataHandler

stream = open("CONFIG.yaml", "r")
opts = yaml.load(stream, Loader=yaml.Loader)

dh = DataHandler(opts)

for database in dh.opts["data"]["databases"]:
    # print(dh.get_database_paths2(database, "train"))
    dh.create_dataset(database)

# dh.create_datasets()
