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

    # def perform_evaluation(
    #     self, evaluator, evaluation_data, scenario_infos, scenario_opts
    # ):
    #     eval_result = {}

    #     if self.opts.get("events_only", False):
    #         print(
    #             "\033[92m"
    #             + "Getting events for model {0} on dataset {1} with evaluator {2}".format(
    #                 scenario_infos["model"],
    #                 scenario_infos["database"],
    #                 scenario_infos["evaluator"],
    #             )
    #             + "\033[0m"
    #         )
    #         eval_result["events"] = evaluator.get_events(
    #             evaluation_data, scenario_opts["evaluator_opts"]
    #         )
    #         eval_result["conf"] = dict(scenario_infos, **scenario_opts)
    #     else:
    #         print(
    #             "\033[92m"
    #             + "Evaluating model {0} on test dataset {1} with evaluator {2}".format(
    #                 scenario_infos["model"],
    #                 scenario_infos["database"],
    #                 scenario_infos["evaluator"],
    #             )
    #             + "\033[0m"
    #         )

    #         start = time.time()
    #         eval_result = evaluator.run_evaluation(
    #             evaluation_data, scenario_opts["evaluator_opts"], scenario_infos
    #         )
    #         end = time.time()
    #         if eval_result:
    #             eval_result["stats"]["PR_curve"] = scenario_opts["evaluator_opts"].get(
    #                 "do_PR_curve", False
    #             )
    #             eval_result["stats"]["duration"] = round(end - start, 2)

    #             eval_result["stats"] = pd.concat(
    #                 [
    #                     pd.DataFrame([scenario_infos]),
    #                     eval_result["stats"].assign(
    #                         **{key: str(value) for key, value in scenario_opts.items()}
    #                     ),
    #                 ],
    #                 axis=1,
    #             )
    #         return eval_result
