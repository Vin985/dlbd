from abc import abstractmethod

import pandas as pd
from plotnine import aes, element_text, geom_line, ggplot, ggtitle, theme, theme_classic

from mouffet.utils.common import (
    deep_dict_update,
    expand_options_dict,
    listdict2dictlist,
)

from mouffet.evaluation.evaluator import Evaluator


class SongDetectorEvaluator(Evaluator):

    EVENTS_COLUMNS = {
        "index": "event_id",
        "event_index": "event_index",
        "recording_id": "recording_id",
        "start": "event_start",
        "end": "event_end",
        "event_duration": "event_duration",
    }
    TAGS_COLUMNS_RENAME = {"id": "tag_id"}

    DEFAULT_ACTIVITY_THRESHOLD = 0.85

    DEFAULT_PR_CURVE_OPTIONS = {
        "variable": "activity_threshold",
        "values": {"end": 1, "start": 0, "step": 0.05},
    }

    def __init__(self):
        pass

    @abstractmethod
    def get_events(self, predictions, options, *args, **kwargs):
        pass

    def run_evaluation(self, predictions, tags, options):
        if options.get("do_PR_curve", False):
            return self.get_PR_curve(predictions, tags, options)
        else:
            return self.evaluate_scenario(predictions, tags, options)

    def evaluate_scenario(self, predictions, tags, options):
        res = self.evaluate(predictions, tags, options)
        res["stats"]["options"] = str(options)
        return res

    @abstractmethod
    def evaluate(self, predictions, tags, options):
        return {"stats": None, "matches": None}

    def get_PR_scenarios(self, options):
        opts = deep_dict_update(
            self.DEFAULT_PR_CURVE_OPTIONS, options.pop("PR_curve", {})
        )
        options[opts["variable"]] = opts["values"]
        scenarios = expand_options_dict(options)
        return scenarios

    def get_PR_curve(self, predictions, tags, options):
        scenarios = self.get_PR_scenarios(options)
        tmp = []
        for scenario in scenarios:
            tmp.append(self.evaluate_scenario(predictions, tags, scenario))

        res = listdict2dictlist(tmp)
        res["matches"] = pd.concat(res["matches"])
        res["stats"] = pd.concat(res["stats"])
        # res["options"] = pd.concat(res["options"])
        res["plots"] = listdict2dictlist(res.get("plots", []))
        if options.get("draw_plots", True):
            res = self.plot_PR_curve(res, options)
        return res

    def draw_plots(self, options, **kwargs):
        return None

    def plot_PR_curve(self, results, options):
        PR_df = results["stats"]

        plt = (
            ggplot(
                data=PR_df,
                mapping=aes(
                    x=options.get("PR_curve_x", "recall"),
                    y=options.get("PR_curve_y", "precision"),
                ),
            )
            + geom_line()
            + theme_classic()
            + theme(
                plot_title=element_text(
                    weight="bold", size=14, margin={"t": 10, "b": 10}
                ),
                figure_size=(20, 10),
                text=element_text(size=12, weight="bold"),
            )
            + ggtitle(
                (
                    "Precision/Recall curve for model {}, database {}, class {}\n"
                    + "with detector options {}"
                ).format(
                    options["scenario_info"]["model"],
                    options["scenario_info"]["database"],
                    options["scenario_info"]["class"],
                    options,
                )
            )
        )

        # plt = PR_df.plot(
        #     "recall_sample", "precision", figsize=(20, 16), fontsize=26
        # ).get_figure()
        # plt.savefig("test_arctic.pdf")
        results["plots"].update({"PR_curve": plt})
        return results

