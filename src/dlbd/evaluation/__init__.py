from mouffet.evaluation import EVALUATORS
from .evaluators import (
    StandardEvaluator,
    SubsamplingEvaluator,
    DirectEvaluator,
)

EVALUATORS.register_evaluators(
    [StandardEvaluator, SubsamplingEvaluator, DirectEvaluator]
)
