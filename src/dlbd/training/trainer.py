import datetime

import yaml

from ..data import utils


class Trainer:
    def __init__(self, opts, data_handler=None, model=None):
        self.opts = opts
        self.model = model
        self.data_handler = data_handler
        self.results_dir = ""

    def train_model(self):
        if not self.data_handler:
            raise AttributeError(
                "An instance of class DataHandler must be provided in data_handler"
                + "attribute or at class initialisation"
            )
        if not self.model:
            raise AttributeError("No model found")
        self.data_handler.check_datasets()
        self.save_params()
        self.model.train(
            self.data_handler.load_data("training"),
            self.data_handler.load_data("validation"),
        )
        self.model.save_weights(self.results_dir)

    def save_params(self):
        if not self.results_dir:
            results_dir = (
                self.opts["model"]["model_dir"]
                + self.model.NAME
                + "/"
                + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                + "/"
            )
            utils.force_make_dir(results_dir)
            self.results_dir = results_dir
        # sys.stdout = ui.Logger(logging_dir + "log.txt")

        with open(self.results_dir + "network_opts.yaml", "w") as f:
            yaml.dump(self.opts, f, default_flow_style=False)
