from mouffet.evaluation.evaluation_handler import EvaluationHandler

from ..data import AudioDataHandler
from . import predictions


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

    def classify_database(self, model, database, db_type="test"):

        db = self.data_handler.load_dataset(
            database,
            db_type,
            load_opts={"file_types": ["spectrograms", "infos"]},
        )
        data = list(zip(db["spectrograms"], db["infos"]))

        preds, infos = predictions.classify_elements(data, model)
        infos["database"] = database.name

        return preds, infos

    def get_predictions_dir(self, model_opts, database):
        preds_dir = super().get_predictions_dir(model_opts, database)
        preds_dir /= self.data_handler.get_spectrogram_subfolder_path(database)
        return preds_dir
