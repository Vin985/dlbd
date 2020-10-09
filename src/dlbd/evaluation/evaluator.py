#%%
from pathlib import Path

import feather
import numpy as np
import pandas as pd

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
            self._test_data = self.data_handler.load_data("test")
        return self._test_data

    # def get_model(self, model_opts):
    #     stream = open(model_opts["options_file"], "r")
    #     model_options = yaml.load(stream, Loader=yaml.Loader) or {}
    #     model_options["resample"] = False
    #     model_options["remove_noise"] = False
    #     weight_path = model_opts["weight_path"]

    #     model = CityNetClassifier1(model_options, weight_path)
    #     return model

    # def get_model_tf2(self, model_opts):
    #     stream = open(model_opts["options_file"], "r")
    #     model_options = yaml.load(stream, Loader=yaml.Loader) or {}
    #     model_options["resample"] = None
    #     model_options["remove_noise"] = False
    #     weight_path = model_opts["weight_path"]

    #     # model = CityNetClassifier1(model_options, weight_path)
    #     model = CityNetTF2(self, model_options)
    #     model.model.load_weights(weight_path)
    #     return model

    # def get_predictions(recordings, model, hop_length=HOP_LENGTH):
    #     res = []
    #     for recording in recordings.itertuples():
    #         preds = []
    #         res_df = pd.DataFrame()
    #         # TODO: see if we can optimize with the recording object
    #         try:
    #             (preds, sr) = model.classify(recording.path)
    #             # print(tmp)
    #             # preds = tmp["preds"]
    #             # sr = tmp["sr"]
    #             len_in_s = preds.shape[0] * hop_length / sr
    #             timeseq = np.linspace(0, len_in_s, preds.shape[0])
    #             res_df = pd.DataFrame(
    #                 {"recording_id": recording.id, "time": timeseq, "activity": preds}
    #             )
    #         except Exception:
    #             print("Error classifying recording: ", recording.path)
    #             print(traceback.format_exc())
    #         res.append(res_df)
    #     res = pd.concat(res)
    #     return res

    # def evaluate_model(self, model, detector, recordings=None):
    #     if not os.path.exists(model_options["save_dest"]):
    #         if model_options.get("is_tf2", False):
    #             model = get_model_tf2(model_options)
    #         else:
    #             model = get_model(model_options)
    #         predictions = get_predictions(recordings, model)
    #         predictions.reset_index(drop=True).to_feather(model_options["save_dest"])
    #     else:
    #         predictions = feather.read_dataframe(model_options["save_dest"])
    #     res = detector.evaluate(predictions, tags, {})
    #     return res

    def get_recordings(self, database):
        pass

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

    def classify_test_data(self, model_opts, version):
        model = self.get_model(model_opts, version)
        specs, _, infos = self.test_data
        res = []
        for i, spec in enumerate(specs):
            preds = model.classify_spectrogram(spec)
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
            # if i > 10:
            #     break
        predictions = pd.concat(res)
        predictions = predictions.astype({"recording_path": "category"})
        # TODO: currently lacking correspondance between file name and spectrogram/labels
        return predictions

    def get_predictions(self, model_opts, version):
        preds_dir = self.get_option("predictions_dir", model_opts)
        if not preds_dir:
            raise AttributeError(
                "Please provide a directory where to save the predictions using"
                + " the predictions_dir option in the config file"
            )
        file_name = (
            "predictions_" + model_opts["name"] + "_v" + str(version) + ".feather"
        )
        pred_file = Path(preds_dir) / file_name
        if not model_opts.get("reclassify", False) and pred_file.exists():
            predictions = feather.read_dataframe(pred_file)
        else:
            predictions = self.classify_test_data(model_opts, version)
            pred_file.parent.mkdir(parents=True, exist_ok=True)
            feather.write_dataframe(predictions, pred_file)
        return predictions

    def prepare_tags(self, tags):
        tags = pd.concat(tags)
        tags = tags.astype({"recording_path": "category"})
        tags["tag_duration"] = tags["tag_end"] - tags["tag_start"]
        tags.reset_index(inplace=True)
        tags.rename(columns={"index": "tag_index"}, inplace=True)
        tags.reset_index(inplace=True)
        tags.rename(columns={"index": "id"}, inplace=True)
        return tags

    def get_tags(self):
        preds_dir = self.opts.get("predictions_dir", ".")
        if not preds_dir:
            raise AttributeError(
                "Please provide a directory where to save the predictions using"
                + " the predictions_dir option in the config file"
            )
        file_name = "test_tags.feather"
        tags_file = Path(preds_dir) / file_name
        if tags_file.exists():
            tags = feather.read_dataframe(tags_file)
        else:
            __, tag_list, _ = self.test_data
            tags = self.prepare_tags(tag_list)
            feather.write_dataframe(tags, tags_file)
        return tags

    def evaluate(self, models=None, recordings=None):
        self.data_handler.check_datasets()
        tags = self.get_tags()
        tags = tags.rename(columns={"recording_path": "recording_id"})
        models = models or self.opts["models"]
        detector_opts = self.opts
        for model_opts in models:
            for version in model_opts["versions"]:
                preds = self.get_predictions(model_opts, version)
                preds = preds.rename(columns={"recording_path": "recording_id"})
                for detector_opts in self.opts["detectors"]:
                    detector = DETECTORS[detector_opts["type"]]
                    detector.evaluate(preds, tags, detector_opts)

