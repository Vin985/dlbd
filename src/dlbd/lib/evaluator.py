#%%
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import feather
import pandas as pd
import plotnine
from plotnine.labels import ggtitle

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
        return (
            database["name"]
            + "_"
            + model_opts["name"]
            + "_v"
            + str(version)
            + ".feather"
        )

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

    def get_tags(self):
        return self.data_handler.load_datasets(
            "test", load_opts={"file_types": ["tags"]}
        )

    def evaluate(self, models=None):
        self.data_handler.check_datasets()
        stats = []
        plots = []
        class_type = self.data_handler.opts["class_type"]
        all_tags = self.get_tags()

        for database in self.data_handler.opts["databases"]:
            if "test" in self.data_handler.get_db_option(
                "db_types", database, self.data_handler.DB_TYPES
            ):
                tags = all_tags[database["name"]]
                models = models or self.opts["models"]
                detector_opts = self.opts

                for model_opts in models:
                    for version in model_opts["versions"]:
                        model_name = model_opts["name"] + "_v" + str(version)
                        preds = self.get_predictions(model_opts, version, database)
                        preds = preds.rename(columns={"recording_path": "recording_id"})
                        for detector_opts in self.opts["detectors"]:
                            detector = self.DETECTORS.get(detector_opts["type"], None)
                            if not detector:
                                print(
                                    "Detector {} not found. Please make sure this detector exists."
                                    + "Skipping."
                                )
                                continue
                            print(
                                "\033[92m"
                                + "Evaluating model {0} on test dataset {1}".format(
                                    model_name, database["name"]
                                )
                                + "\033[0m"
                            )
                            model_stats = detector.evaluate(preds, tags, detector_opts)
                            model_stats["stats"]["database"] = database["name"]
                            model_stats["stats"]["model"] = model_name
                            model_stats["stats"]["type"] = str(detector_opts)
                            model_stats["stats"]["tag_class"] = class_type
                            stats.append(pd.Series(model_stats["stats"]))
                            plt = model_stats.get("tag_repartition", None)
                            if plt:
                                plt += ggtitle(
                                    "Tag repartition for model {}, database {}\nwith detector options {}".format(
                                        model_name,
                                        database["name"],
                                        model_stats["stats"]["type"],
                                    )
                                )
                                plots.append(plt)
        stats_df = pd.DataFrame(stats)
        if self.opts.get("save_results", True):
            time = datetime.now()
            res_dir = Path(self.opts.get("evaluation_dir", ".")) / time.strftime(
                "%Y%m%d"
            )
            prefix = time.strftime("%H%M%S")
            eval_id = self.opts.get("id", "")
            stats_df.to_csv(
                str(
                    file_utils.ensure_path_exists(
                        res_dir
                        / (
                            "_".join(
                                filter(None, [prefix, class_type, eval_id, "stats.csv"])
                            )
                        ),
                        is_file=True,
                    )
                ),
            )
            if plots:
                plotnine.save_as_pdf_pages(
                    plots,
                    res_dir
                    / (
                        "_".join(
                            filter(
                                None,
                                [prefix, class_type, eval_id, "tag_repartition.pdf"],
                            )
                        )
                    ),
                )
        return stats_df

