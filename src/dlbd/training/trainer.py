class Trainer:
    def __init__(self, opts, data_handler=None, model=None):
        self.opts = opts
        self.model = model
        self.data_handler = data_handler

    def train_model(self):
        if not self.data_handler:
            raise AttributeError(
                "An instance of class DataHandler must be provided in data_handler"
                + "attribute or at class initialisation"
            )
        if not self.model:
            raise AttributeError("No model found")
        self.data_handler.check_datasets()
        self.model.train(
            self.data_handler.load_data("training"),
            self.data_handler.load_data("validation"),
        )

