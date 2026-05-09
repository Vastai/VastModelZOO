from .text_detector.detector import Detector
from .text_recognizer.recognizer import Recognizer
from .pipeline import OCRPipeline
from .schema import OCRResult
from .config import OCRConfig, DetectorConfig, RecognizerConfig, VisualizeConfig
from .eval import TextDetMetric, TextRecMetric

__version__ = "1.0.0"
__all__ = [
    "OCRPipeline",
    "OCRResult",
    "Detector",
    "Recognizer",
    "OCRConfig",
    "DetectorConfig",
    "RecognizerConfig",
    "VisualizeConfig",
    "TextDetMetric",
    "TextRecMetric"
]
