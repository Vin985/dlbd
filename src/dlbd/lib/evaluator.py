#%%
from abc import ABC, abstractmethod
from datetime import datetime
from itertools import product
from pathlib import Path

import feather
import pandas as pd
import plotnine
from plotnine.labels import ggtitle
from dlbd.options.database_options import DatabaseOptions

from dlbd.options.model_options import ModelOptions

from ..utils import common as common_utils
from ..utils import file as file_utils
from ..utils.model_handler import ModelHandler


class Evaluator(ModelHandler, ABC):

    DETECTORS = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_model(self, model_opts, version):
        old_opts = file_utils.load_config(
            Path(self.get_option("weights_dir", model_opts))
            / model_opts["name"]
            / str(version)
            / "network_opts.yaml"
        )
        if "options" in model_opts:
            common_utils.deep_dict_update(old_opts, model_opts["options"])
        model = self.get_model_instance(model_opts, old_opts, version)
        model.load_weights()
        return model

    @abstractmethod
    def classify_test_data(self, model, database):
        pass

    def get_predictions_dir(self, model_opts, database):
        preds_dir = self.get_option("predictions_dir", model_opts)
        if not preds_dir:
            raise AttributeError(
                "Please provide a directory where to save the predictions using"
                + " the predictions_dir option in the config file"
            )
        return Path(preds_dir)

    def get_predictions_file_name(self, model_opts, version, database):
        return database.name + "_" + model_opts.model_id + ".feather"

    def get_predictions(self, model_opts, version, database):
        preds_dir = self.get_predictions_dir(model_opts, database)
        file_name = self.get_predictions_file_name(model_opts, version, database)
        pred_file = preds_dir / file_name
        if not model_opts.get("reclassify", False) and pred_file.exists():
            predictions = feather.read_dataframe(pred_file)
        else:
            model = self.get_model(model_opts, version)
            predictions = self.classify_test_data(model, database)
            pred_file.parent.mkdir(parents=True, exist_ok=True)
            feather.write_dataframe(predictions, pred_file)
        return predictions

    # def get_tags(self):
    #     return self.data_handler.load_datasets(
    #         "test", load_opts={"file_types": ["tags"]}
    #     )

    # def load_databases_options(self):
    #     res = []
    #     for database in self.opts["databases"]:
    #         original_opts = self.data_handler.get_database_options(database["name"])
    #         if "test" in self.data_handler.get_db_option(
    #             "db_types", database, self.data_handler.DB_TYPES
    #         ):
    #             new_opts = common_utils.deep_dict_update(
    #                 original_opts, database, copy=True
    #             )
    #             res.append(new_opts)
    #     return res

    def get_detector(self, detector_opts):
        detector = self.DETECTORS.get(detector_opts["type"], None)
        if not detector:
            print(
                "Detector {} not found. Please make sure this detector exists."
                + "Skipping."
            )
            return None
        return detector

    # def run_detector(self, predictions, tags, detector_opts):

    #     tags = self.data_handler.load_dataset(
    #         database, "test", load_opts={"file_types": detector.TAG_TYPES}
    #     )
    #     model_stats = detector.evaluate(predictions, tags, detector_opts)
    #     return model_stats

    # def evaluate_model(self, model_opts, database, tags, class_type):
    #     stats = []
    #     for version in model_opts["versions"]:
    #         model_name = model_opts["name"] + "_v" + str(version)
    #         model_id = model_name
    #         mid = model_opts.get("id", "")
    #         if mid:
    #             model_id = model_id + "_" + mid
    #         model_opts["model_id"] = model_id
    #         preds = self.get_predictions(model_opts, version, database)
    #         preds = preds.rename(columns={"recording_path": "recording_id"})
    #         for detector_opts in self.opts["detectors"]:
    #             stats_infos = {
    #                 "database": database["name"],
    #                 "model": model_id,
    #                 "class": class_type,
    #                 "detector_opts": str(detector_opts),
    #                 "database_opts": str(database.updated_opts),
    #                 "model_opts": str(model_opts),
    #             }
    #             print(
    #                 "\033[92m"
    #                 + "Evaluating model {0} on test dataset {1}".format(
    #                     model_name, database["name"]
    #                 )
    #                 + "\033[0m"
    #             )
    #             model_stats = self.run_detector(preds, tags, detector_opts)
    #             model_stats["stats"].update(stats_infos)
    #             plt = model_stats.get("tag_repartition", None)
    #             if plt:
    #                 plt += ggtitle(
    #                     (
    #                         "Tag repartition for model {}, database {}, class {}\n"
    #                         + "with detector options {}"
    #                     ).format(
    #                         model_id, database["name"], class_type, detector_opts,
    #                     )
    #                 )
    #             stats.append(model_stats)
    #     return stats

    # def evaluate_database(self, database, db_tags, models, detectors):
    #     stats = []
    #     class_type = self.data_handler.get_db_option("class_type", database)
    #     tags = db_tags[database["name"]]
    #     models = models or self.opts["models"]

    #     for model_opts in models:
    #         m_stats = self.evaluate_model(model_opts, database, tags, class_type)
    #         stats += m_stats
    #     return stats

    def consolidate_stats(self, stats):
        tmp_stats, plots = [], []
        for stat in stats:
            tmp_stats.append(pd.Series(stat["stats"]))
            plt = stat.get("tag_repartition", None)
            if plt:
                plots.append(plt)
        stats_df = pd.DataFrame(tmp_stats)
        return stats_df, plots

    def save_results(self, stats):
        stats_df, plots = self.consolidate_stats(stats)
        time = datetime.now()
        res_dir = Path(self.opts.get("evaluation_dir", ".")) / time.strftime("%Y%m%d")
        prefix = time.strftime("%H%M%S")
        eval_id = self.opts.get("id", "")
        stats_df.to_csv(
            str(
                file_utils.ensure_path_exists(
                    res_dir / ("_".join(filter(None, [prefix, eval_id, "stats.csv"]))),
                    is_file=True,
                )
            ),
        )
        if plots:
            plotnine.save_as_pdf_pages(
                plots,
                res_dir
                / ("_".join(filter(None, [prefix, eval_id, "tag_repartition.pdf"],))),
            )

    def expand_scenarios(self, option_type):
        elements = self.opts[option_type]
        db_opts = []
        for element in elements:
            if "scenarios" in element:
                clean = dict(element)
                clean.pop("scenarios")
                for opts in common_utils.expand_options_dict(element["scenarios"]):
                    res = dict(clean)
                    res = common_utils.deep_dict_update(res, opts, copy=True)
                    db_opts.append(res)
            else:
                db_opts.append(dict(element))
        return db_opts

    def load_databases_options(self):
        return self.expand_scenarios("databases")

    def load_model_options(self):
        return self.expand_scenarios("models")

    def load_detector_options(self):
        return self.expand_scenarios("detectors")

    def load_scenarios(self):
        db_opts = self.load_databases_options()
        model_opts = self.load_model_options()
        detector_opts = self.load_detector_options()
        res = product(db_opts, model_opts, detector_opts)
        return list(res)

    def load_tags(self, database, types):
        return self.data_handler.load_dataset(
            database, "test", load_opts={"file_types": types}
        )

    def evaluate_scenario(self, opts):
        db_opts, model_opts, detector_opts = opts
        model_opts = ModelOptions(model_opts)
        database = self.data_handler.duplicate_database(db_opts)
        model_stats = {}
        if database and database.has_type("test"):
            self.data_handler.check_dataset(database, ["test"])
            preds = self.get_predictions(model_opts, model_opts.version, database)
            preds = preds.rename(columns={"recording_path": "recording_id"})
            stats_infos = {
                "database": database.name,
                "model": model_opts.model_id,
                "class": database.class_type,
            }
            stats_opts = {
                "detector_opts": str(detector_opts),
                "database_opts": str(database.updated_opts),
                "model_opts": str(model_opts),
            }
            print(
                "\033[92m"
                + "Evaluating model {0} on test dataset {1}".format(
                    model_opts.model_name, database.name
                )
                + "\033[0m"
            )
            detector = self.get_detector(detector_opts)
            if detector:
                tags = self.load_tags(database, detector.REQUIRES)
                model_stats = detector.evaluate(preds, tags, detector_opts)
                stats_infos.update(model_stats["stats"])
                stats_infos.update(stats_opts)
                model_stats["stats"] = stats_infos
                plt = model_stats.get("tag_repartition", None)
                if plt:
                    plt += ggtitle(
                        (
                            "Tag repartition for model {}, database {}, class {}\n"
                            + "with detector options {}"
                        ).format(
                            model_opts.model_id,
                            database.name,
                            database.class_type,
                            detector_opts,
                        )
                    )
        return model_stats

    def evaluate(self):
        scenarios = self.load_scenarios()
        stats = [self.evaluate_scenario(scenario) for scenario in scenarios]

        if self.opts.get("save_results", True):
            self.save_results(stats)
        return stats

