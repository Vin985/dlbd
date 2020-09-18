from ..data.data_handler import DataHandler


class Trainer:
    def __init__(self, opts, model=None):
        self.opts = opts
        self.model = model

        self.data_handler = DataHandler(opts)

    def train_model(self):
        if not self.model:
            raise AttributeError("No model found")
        self.model.train(
            self.data_handler.load_data("train"), self.data_handler.load_data("test")
        )
