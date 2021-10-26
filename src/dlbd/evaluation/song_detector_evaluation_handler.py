import numpy as np
import pandas as pd
from mouffet.evaluation.evaluation_handler import EvaluationHandler

from ..data.audio_data_handler import AudioDataHandler
from ..training.spectrogram_sampler import SpectrogramSampler
from .evaluators.standard_evaluator import StandardEvaluator
from .evaluators.subsampling_evaluator import SubsamplingEvaluator


class SongDetectorEvaluationHandler(EvaluationHandler):

    DATA_HANDLER_CLASS = AudioDataHandler

    EVENTS_COLUMNS = {
        "index": "event_id",
        "event_index": "event_index",
        "recording_id": "recording_id",
        "start": "event_start",
        "end": "event_end",
        "event_duration": "event_duration",
    }
    TAGS_COLUMNS_RENAME = {"id": "tag_id"}

    EVALUATORS = {
        "standard": StandardEvaluator(),
        "subsampling": SubsamplingEvaluator(),
    }

    @staticmethod
    def classify_element(model, spectrogram, info, sampler):
        preds = model.classify((spectrogram, info), sampler)
        pix_in_sec = model.opts.get("pixels_per_sec", 20)
        len_in_s = preds.shape[0] / pix_in_sec
        timeseq = np.linspace(0, len_in_s, preds.shape[0])
        res_df = pd.DataFrame(
            {
                "recording_path": str(info["file_path"]),
                "time": timeseq,
                "activity": preds,
            }
        )
        return res_df

    def classify_test_data(self, model, database):
        test_data = self.data_handler.load_dataset(
            database, "test", load_opts={"file_types": ["spectrograms", "infos"]}
        )
        res = []

        test_sampler = SpectrogramSampler(model.opts, balanced=False)
        test_sampler.opts["do_augmentation"] = False
        # test_sampler.opts["batch_size"] = 256
        for i, spec in enumerate(test_data["spectrograms"]):
            res_df = self.classify_element(
                model, spec, test_data["infos"][i], test_sampler
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

    def load_tags(self, database, types):
        return self.data_handler.load_dataset(
            database,
            "test",
            load_opts={
                "file_types": types,
                "onload_callbacks": {"tags_df": self.prepare_tags},
            },
        )
