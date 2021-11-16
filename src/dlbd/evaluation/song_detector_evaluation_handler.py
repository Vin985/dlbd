import time

import numpy as np
import pandas as pd
from mouffet.evaluation.evaluation_handler import EvaluationHandler

from dlbd.evaluation.evaluators.citynet_evaluator import CityNetEvaluator


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
        "citynet": CityNetEvaluator(),
    }

    @staticmethod
    def classify_element(model, spectrogram, info, sampler):
        preds = model.classify((spectrogram, info), sampler)
        len_in_s = info["length"] / info["sample_rate"]
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
        infos = {}
        res = []
        total_audio_duration = 0

        test_data = self.data_handler.load_dataset(
            database, "test", load_opts={"file_types": ["spectrograms", "infos"]}
        )
        infos["database"] = database.name
        infos["n_files"] = len(test_data["spectrograms"])

        test_sampler = SpectrogramSampler(model.opts, balanced=False)
        test_sampler.opts["do_augmentation"] = False
        start = time.time()
        # test_sampler.opts["batch_size"] = 256
        for i, spec in enumerate(test_data["spectrograms"]):
            audio_infos = test_data["infos"][i]
            total_audio_duration += audio_infos["length"] / audio_infos["sample_rate"]
            res_df = self.classify_element(model, spec, audio_infos, test_sampler)
            res.append(res_df)

        end = time.time()

        infos["global_duration"] = round(end - start, 2)
        infos["total_audio_duration"] = round(total_audio_duration, 2)
        infos["average_time_per_min"] = round(
            infos["global_duration"] / (total_audio_duration / 60), 2
        )
        infos["average_time_per_file"] = round(
            infos["global_duration"] / infos["n_files"], 2
        )
        infos["spectrogram_overlap"] = test_sampler.opts["overlap"]

        predictions = pd.concat(res)
        predictions = predictions.astype({"recording_path": "category"})
        return predictions, infos

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
