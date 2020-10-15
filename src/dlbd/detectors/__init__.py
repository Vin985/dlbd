from .standard_detector import StandardDetector
from .subsampling_detector import SubsamplingDetector

from .citynet_detector import CityNetDetector

DETECTORS = {
    "standard": StandardDetector(),
    "subsampling": SubsamplingDetector(),
    "citynet": CityNetDetector(),
}

