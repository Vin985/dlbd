from ..utils import file as file_utils

from ..data.data_handler import DataHandler


class Trainer:
    def __init__(self, opts=None, opts_path="", dh=None, model=None, **kwargs):
        if not opts:
            if opts_path:
                opts = file_utils.load_config(opts_path)
            else:
                raise AttributeError(
                    "You should provide either an options dict via the opts attribute"
                    + "or the path to the config file via opts_path"
                )
        self.opts = opts
        self._model = None
        self._data_handler = None
        if model:
            self.model = model
        if not dh:
            dh = self.create_data_handler(**kwargs)
        self.data_handler = dh

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model and not model.opts:
            model.opts = self.opts
        self._model = model

    @property
    def data_handler(self):
        return self._data_handler

    @data_handler.setter
    def data_handler(self, dh):
        if dh and not dh.opts:
            dh.opts = self.opts
        self._data_handler = dh

    def create_data_handler(self, split_funcs=None):
        data_opts_path = self.opts.get("data_config", "")
        if not data_opts_path:
            raise Exception("A path to the data config file must be provided")
        data_opts = file_utils.load_config(data_opts_path)
        dh = DataHandler(data_opts, split_funcs=split_funcs)
        return dh

    def train_model(self):
        if not self.data_handler:
            raise AttributeError(
                "An instance of class DataHandler must be provided in data_handler"
                + "attribute or at class initialisation"
            )
        if not self.model:
            raise AttributeError("No model found")
        self.data_handler.check_datasets()
        training_data = self.model.prepare_data(self.data_handler.load_data("training"))
        validation_data = self.model.prepare_data(
            self.data_handler.load_data("validation")
        )
        self.model.train(training_data, validation_data)

