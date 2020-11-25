from importlib import import_module


from ..data.data_handler import DataHandler
from ..models.dl_model import DLModel
from ..utils import file as file_utils


class ModelHandler:
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

    def create_data_handler(self, dh_class=None, split_funcs=None):
        data_opts_path = self.opts.get("data_config", "")
        if not data_opts_path:
            raise Exception("A path to the data config file must be provided")
        data_opts = file_utils.load_config(data_opts_path)
        if dh_class:
            dh = dh_class(data_opts, split_funcs=split_funcs)
        else:
            dh = DataHandler(data_opts, split_funcs=split_funcs)
        return dh

    def get_model_instance(self, model, model_opts, version=None):
        if isinstance(model, DLModel):
            return model
        if not isinstance(model, dict):
            raise ValueError(
                "Model should either be an instance of dlbd.models.DLModel or a dict"
            )

        pkg = import_module(model["package"])
        cls = getattr(pkg, model["name"])
        return cls(model_opts, version=version)

    def get_option(self, name, group, default=""):
        return group.get(name, self.opts.get(name, default))

