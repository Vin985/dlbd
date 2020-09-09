from .standard_detector import StandardDetector
from .subsampling_detector import SubsamplingDetector

DETECTORS = {
    "standard": StandardDetector(),
    "subsampling": SubsamplingDetector(),
}

