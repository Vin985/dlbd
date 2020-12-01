import numpy as np
import pandas as pd
from dlbd.training.spectrogram_sampler import SpectrogramSampler

from ..lib.evaluator import Evaluator

# from .citynet_detector import CityNetDetector
from .standard_detector import StandardDetector
from .subsampling_detector import SubsamplingDetector


class AudioDetectorEvaluator(Evaluator):

    DETECTORS = {
        "standard": StandardDetector(),
        "subsampling": SubsamplingDetector(),
        # "citynet": CityNetDetector(),
    }

    def classify_test_data(self, model, database):
        test_data = self.data_handler.load_dataset(
            database, "test", load_opts={"file_types": ["spectrograms", "infos"]}
        )
        res = []

        test_sampler = SpectrogramSampler(model.opts, balanced=False)
        test_sampler.opts["do_augmentation"] = False
        # test_sampler.opts["batch_size"] = 256
        for i, spec in enumerate(test_data["spectrograms"]):
            preds = model.classify_spectrogram(spec, test_sampler)
            info = test_data["infos"][i]
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

    def prepare_tags(self, tags):
        tags = tags.astype({"recording_path": "category"})
        tags["tag_duration"] = tags["tag_end"] - tags["tag_start"]
        tags.reset_index(inplace=True)
        tags.rename(
            columns={"index": "tag_index", "recording_path": "recording_id"},
            inplace=True,
        )
        tags.reset_index(inplace=True)
        tags.rename(columns={"index": "id"}, inplace=True)
        return tags

    def get_predictions_dir(self, model_opts, database):
        preds_dir = super().get_predictions_dir(model_opts, database)
        preds_dir /= self.data_handler.get_spectrogram_subfolder_path(database)
        return preds_dir

    def get_tags(self):
        load_opts = {
            "file_types": ["tags_df", "tags_linear"],
            "onload_callbacks": {"tags_df": self.prepare_tags},
        }
        tags = self.data_handler.load_datasets(
            "test", load_opts=load_opts, by_dataset=True,
        )
        return tags

