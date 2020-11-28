#%%
from pathlib import Path

import feather
import numpy as np
import pandas as pd
import plotnine
from plotnine.labels import ggtitle
from dlbd import data
from dlbd.data import tag_manager
from dlbd.training.spectrogram_sampler import SpectrogramSampler

from ..detectors import DETECTORS
from ..utils import file as file_utils
from ..utils.model_handler import ModelHandler


class Evaluator(ModelHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_data = None

    @property
    def test_data(self):
        if not self._test_data:
            self._test_data = self.data_handler.load_data("test", by_dataset=True)
        return self._test_data

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

    def classify_test_data(self, model, database):
        specs, _, infos = self.test_data[database]
        res = []

        test_sampler = SpectrogramSampler(model.opts, balanced=False)
        test_sampler.opts["do_augmentation"] = False
        test_sampler.opts["batch_size"] = 256
        for i, spec in enumerate(specs):
            preds = model.classify_spectrogram(spec, test_sampler)
            info = infos[i]
            len_in_s = (
                preds.shape[0]
                * info["spec_opts"]["hop_length"]
                / info["spec_opts"]["sr"]
            )
            timeseq = np.linspace(0, len_in_s, preds.shape[0])
            res_df = pd.DataFrame(
                {
                    "recording_path": str(info["file_path"]),
                    "time": timeseq,
                    "activity": preds,
                }
            )
            res.append(res_df)
        predictions = pd.concat(res)
        predictions = predictions.astype({"recording_path": "category"})
        return predictions

    def get_predictions(self, model_opts, version, database):
        preds_dir = self.get_option("predictions_dir", model_opts)
        if not preds_dir:
            raise AttributeError(
                "Please provide a directory where to save the predictions using"
                + " the predictions_dir option in the config file"
            )
        file_name = (
            "predictions_"
            + database
            + "_"
            + model_opts["name"]
            + "_v"
            + str(version)
            + ".feather"
        )
        pred_file = Path(preds_dir) / file_name
        if not model_opts.get("reclassify", False) and pred_file.exists():
            predictions = feather.read_dataframe(pred_file)
        else:
            model = self.get_model(model_opts, version)
            predictions = self.classify_test_data(model, database)
            pred_file.parent.mkdir(parents=True, exist_ok=True)
            feather.write_dataframe(predictions, pred_file)
        return predictions

    def prepare_tags(self, tags):
        # tags = pd.concat(tags)
        tags = tags.astype({"recording_path": "category"})
        tags["tag_duration"] = tags["tag_end"] - tags["tag_start"]
        tags.reset_index(inplace=True)
        tags.rename(columns={"index": "tag_index"}, inplace=True)
        tags.reset_index(inplace=True)
        tags.rename(columns={"index": "id"}, inplace=True)
        return tags

    def get_tags(self, database):
        paths = self.data_handler.get_database_paths(database)
        # file_name = database + "_test_tags.feather"
        tags_file = paths["tag_df"]["test"]
        # if tags_file.exists():
        tags = feather.read_dataframe(tags_file)
        classes = self.data_handler.load_classes(database)
        tags = tag_manager.filter_classes(tags, classes)
        tags = self.prepare_tags(tags)
        # else:
        #     __, tag_list, _ = self.test_data[database]
        #     tags = self.prepare_tags(tag_list)
        #     feather.write_dataframe(tags, tags_file)
        return tags

    def evaluate(self, models=None):
        self.data_handler.check_datasets()
        stats = []
        plots = []
        class_type = self.data_handler.opts["class_type"]
        for database in self.data_handler.opts["databases"]:
            if "test" in self.data_handler.get_db_option(
                "db_types", database, self.data_handler.DB_TYPES
            ):
                tags = self.get_tags(database)
                tags = tags.rename(columns={"recording_path": "recording_id"})
                models = models or self.opts["models"]
                detector_opts = self.opts

                for model_opts in models:
                    for version in model_opts["versions"]:
                        model_name = model_opts["name"] + "_v" + str(version)
                        preds = self.get_predictions(
                            model_opts, version, database["name"]
                        )
                        preds = preds.rename(columns={"recording_path": "recording_id"})
                        for detector_opts in self.opts["detectors"]:
                            detector = DETECTORS[detector_opts["type"]]
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
            res_dir = Path(self.opts.get("evaluation_dir", "."))
            stats_df.to_csv(
                str(
                    file_utils.ensure_path_exists(
                        res_dir / (class_type + "_stats_noresampling.csv"),
                        is_file=True,
                    )
                ),
            )
            if plots:
                plotnine.save_as_pdf_pages(
                    plots, res_dir / (class_type + "_tag_repartition3.pdf")
                )
        return stats_df
