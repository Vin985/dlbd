from mouffet.evaluation import EVALUATORS
from .evaluators import (
    StandardEvaluator,
    SubsamplingEvaluator,
    CityNetEvaluator,
)

EVALUATORS.register_evaluators(
    {
        "standard": StandardEvaluator,
        "subsampling": SubsamplingEvaluator,
        "citynet": CityNetEvaluator,
    }
)
