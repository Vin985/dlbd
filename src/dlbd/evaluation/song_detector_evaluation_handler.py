from mouffet.evaluation.evaluation_handler import EvaluationHandler

from ..data import AudioDataHandler
from . import predictions


class SongDetectorEvaluationHandler(EvaluationHandler):

    DATA_HANDLER_CLASS = AudioDataHandler

    def predict_database(self, model, database, db_type="test"):

        db = self.data_handler.load_dataset(
            db_type,
            database,
            load_opts={"file_types": ["spectrograms", "infos"]},
            prepare=True,
            prepare_opts=model.opts.opts,
        )
        data = list(zip(db["spectrograms"], db["infos"]))

        preds, infos = predictions.classify_elements(data, model)
        infos["database"] = database.name

        return preds, infos

    def get_predictions_dir(self, model_opts, database):
        preds_dir = super().get_predictions_dir(model_opts, database)
        preds_dir /= self.data_handler.DATASET(
            "test", database
        ).get_spectrogram_subfolder_path()
        return preds_dir

    def on_get_predictions_end(self, preds):
        preds = preds.rename(columns={"recording_path": "recording_id"})
        return preds

    def get_predictions_file_name(self, model_opts, database):
        return (
            database.name
            + "_"
            + model_opts.model_id
            + "_v"
            + str(model_opts.load_version)
            + "_overlap"
            + str(model_opts.spectrogram_overlap)
            + ".feather"
        )
