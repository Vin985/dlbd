from mouffet.evaluation.evaluation_handler import EvaluationHandler

from ..data import AudioDataHandler
from . import predictions


class SongDetectorEvaluationHandler(EvaluationHandler):

    DATA_HANDLER_CLASS = AudioDataHandler
    PREDICTIONS_STATS_DUPLICATE_COLUMNS = [
        "database",
        "model_id",
        "spectrogram_overlap",
    ]

    def predict_database(self, model, database, db_type="test"):

        db = self.data_handler.load_dataset(
            db_type,
            database,
            load_opts={"file_types": ["spectrograms", "metadata", "spec_opts"]},
            prepare=True,
            prepare_opts=model.opts.opts,
        )
        data = list(zip(db["spectrograms"], db["metadata"]))

        preds, infos = predictions.classify_elements(data, model, db["spec_opts"])
        infos["database"] = database.name

        return preds, infos

    def get_predictions_dir(self, model_opts, database):
        preds_dir = super().get_predictions_dir(model_opts, database)
        preds_dir /= self.data_handler.DATASET(
            "test", database
        ).get_spectrogram_subfolder_path()
        return preds_dir

    def smooth_predictions(self, preds, model_opts):
        factor = model_opts.get("smooth_factor", 5)
        if factor:
            roll = preds["activity"].rolling(factor, center=True)
            preds.loc[:, "activity"] = roll.mean()
        return preds

    def on_get_predictions_end(self, preds, model_opts):
        preds = preds.rename(columns={"recording_path": "recording_id"})
        if model_opts.get("smooth_predictions", False):
            preds = preds.groupby("recording_id").apply(
                self.smooth_predictions, model_opts
            )
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
