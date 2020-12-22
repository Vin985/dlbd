import traceback

from dlbd.options.model_options import ModelOptions

from ..utils import common as common_utils
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
        db_types = [
            self.data_handler.DB_TYPE_TRAINING,
            self.data_handler.DB_TYPE_VALIDATION,
        ]

        self.data_handler.check_datasets(db_types=db_types)
        data = [
            self.model.prepare_data(self.data_handler.load_datasets(db_type))
            for db_type in db_types
        ]
        self.model.train(*data)

    def expand_training_scenarios(self):
        scenarios = []
        if "scenarios" in self.opts:
            clean = dict(self.opts)
            clean.pop("scenarios")
            for scenario in self.opts["scenarios"]:
                for opts in common_utils.expand_options_dict(
                    scenario  # , exclude_expand=["databases_options"]
                ):
                    res = dict(clean)
                    res = common_utils.deep_dict_update(res, opts, copy=True)
                    scenarios.append(res)
        else:
            scenarios.append(dict(self.opts))
        return scenarios

    def load_scenarios(self):
        return self.expand_training_scenarios()

    def get_scenario_databases_options(self, scenario):
        db_opts = []
        opts_update = scenario.get("databases_options", {})
        for db_name in scenario["databases"]:
            db_opt = self.data_handler.update_database(opts_update, db_name)
            if db_opt:
                db_opts.append(db_opt)
        return db_opts

    def train(self):
        if not self.data_handler:
            raise AttributeError(
                "An instance of class DataHandler must be provided in data_handler"
                + "attribute or at class initialisation"
            )
        if not self.model:
            raise AttributeError("No model found")
        db_types = [
            self.data_handler.DB_TYPE_TRAINING,
            self.data_handler.DB_TYPE_VALIDATION,
        ]
        for scenario in self.scenarios:
            try:
                databases = self.get_scenario_databases_options(scenario)
                self.data_handler.check_datasets(databases=databases, db_types=db_types)
                data = [
                    self.model.prepare_data(self.data_handler.load_datasets(db_type))
                    for db_type in db_types
                ]
                print(scenario)
                self.model.opts = ModelOptions(scenario)
                self.model.opts.name = self.model.NAME
                self.model.train(*data)
            except Exception:
                print(traceback.format_exc())
                print("Error training the model for scenario ", scenario)
