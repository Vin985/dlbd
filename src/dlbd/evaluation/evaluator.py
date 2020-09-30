#%%
import os
import traceback
from pathlib import Path
from importlib import import_module

import numpy as np
import pandas as pd
import yaml
import feather

from dlbd.data import data_handler
from dlbd.data.data_handler import DataHandler

from ..detectors import DETECTORS
from ..models.dl_model import DLModel


class Evaluator:
    def __init__(self, opts):
        self.opts = opts
        self.data_handler = DataHandler(opts)

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

    def get_model_class(self, model):
        if isinstance(model, DLModel):
            return model
        if not isinstance(model, dict):
            raise ValueError(
                "Model should either be an instance of dlbd.models.DLModel or a dict"
            )

        pkg = import_module(model["package"])
        cls = getattr(pkg, model["name"])
        return cls(self.opts)

    def get_recordings(self, database):
        pass

    def evaluate(self, models=None, recordings=None):
        models = models or self.opts["models"]

        for model in models:
            model_instance = self.get_model_class(model)

        if not recordings:
            tmp = []
            for database in self.opts["databases"]:
                tmp.append(self.get_recordings(database))
            recordings = pd.concat(tmp)
