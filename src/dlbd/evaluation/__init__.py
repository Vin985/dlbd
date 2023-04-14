from mouffet.evaluation import EVALUATORS
from .evaluators import (
    StandardEvaluator,
    SubsamplingEvaluator,
    DirectEvaluator,
    PresenceEvaluator,
)

EVALUATORS.register_evaluators(
    [StandardEvaluator, SubsamplingEvaluator, DirectEvaluator, PresenceEvaluator]
)
