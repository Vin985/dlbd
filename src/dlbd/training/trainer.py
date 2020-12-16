from ..utils.model_handler import ModelHandler


class Trainer(ModelHandler):
    def train_model(self):
        db_types = [
            self.data_handler.DB_TYPE_TRAINING,
            self.data_handler.DB_TYPE_VALIDATION,
        ]
        if not self.data_handler:
            raise AttributeError(
                "An instance of class DataHandler must be provided in data_handler"
                + "attribute or at class initialisation"
            )
        if not self.model:
            raise AttributeError("No model found")
        self.data_handler.check_datasets(db_types=db_types)
        data = [
            self.model.prepare_data(self.data_handler.load_datasets(db_type))
            for db_type in db_types
        ]
        self.model.train(*data)

