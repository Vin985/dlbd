from ..utils.model_handler import ModelHandler


class Trainer(ModelHandler):
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

