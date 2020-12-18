from ..utils import common as common_utils
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

    def expand_training_scenarios(self):
        scenarios = []
        if "scenarios" in self.opts:
            clean = dict(self.opts)
            clean.pop("scenarios")
            for scenario in self.opts["scenarios"]:
                # TODO: do not expand the dictionaries? Keep the paradigm of 1 scenario, 1 model?
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

    def get_db_opts_to_update(self, opts_update, db_name):
        if isinstance(opts_update, list):
            print("in list")
            for opts in opts_update:
                if opts.get("name", "") == db_name:
                    return opts
        else:
            if opts_update.get("name", "") == db_name:
                return opts_update
        return None

    def get_scenario_databases_options(self, scenario):
        db_opts = []
        opts_update = scenario.get("databases_options", {})
        for db_name in scenario["databases"]:
            db_opt = self.data_handler.update_database(opts_update, db_name)
            print(db_opt)
            # if opts_update:
            #     # new_opts = self.get_db_opts_to_update(opts_update, db_name)
            #     # if new_opts:
            #     db_opt = self.data_handler.duplicate_database(opts_update)
            if db_opt:
                db_opts.append(db_opt)
        return db_opts

    def train(self):
        # print(self.scenarios)
        for scenario in self.scenarios:
            databases = self.get_scenario_databases_options(scenario)
            # print(scenario)

