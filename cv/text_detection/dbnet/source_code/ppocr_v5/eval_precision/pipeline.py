import cv2
import time
import numpy as np
from PIL import Image
from loguru import logger
from typing import List, Union, Optional
from pathlib import Path

from .text_detector.detector import Detector
from .text_recognizer.recognizer import Recognizer
from .schema import OCRResult
from .tools.visualizer import OCRVisualizer
from .config import OCRConfig, DetectorConfig, RecognizerConfig


class OCRPipeline:
    """
    End-to-end OCR Pipeline combining detection and recognition.
    
    Examples:
        # Method 1: From pretrained config
        >>> pipeline = OCRPipeline.from_pretrained("mobile")
        
        # Method 2: From YAML file
        >>> pipeline = OCRPipeline.from_config("config.yaml")
        
        # Method 3: Custom config
        >>> config = OCRConfig(
        ...     det=DetectorConfig(path="det.onnx"),
        ...     rec=RecognizerConfig(path="rec.onnx")
        ... )
        >>> pipeline = OCRPipeline(config)
        
        # Method 4: Direct paths
        >>> pipeline = OCRPipeline.from_model_paths(
        ...     det_model_path="det.onnx",
        ...     rec_model_path="rec.onnx"
        ... )
    """
    
    def __init__(
        self,
        config: OCRConfig,
        det_providers: Optional[List[str]] = None,
        rec_providers: Optional[List[str]] = None,
        visualize: bool = False
    ):
        """
        Initialize OCR Pipeline with configuration.
        
        Args:
            config: OCRConfig object containing all settings.
            det_providers: ONNX Runtime providers for detection.
            rec_providers: ONNX Runtime providers for recognition.
            visualize: Enable visualization of results.
        """
        self.config = config
        self.detector = Detector(config.det, providers=det_providers)
        self.recognizer = Recognizer(config.rec, providers=rec_providers)
        
        if visualize:
            self.visualizer = OCRVisualizer(
                font_path=config.visualize.font_path,
                box_thickness=config.visualize.box_thickness
            )
        else:
            self.visualizer = None
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "mobile",
        det_providers: Optional[List[str]] = None,
        rec_providers: Optional[List[str]] = None,
        visualize: bool = False
    ) -> "OCRPipeline":
        """
        Create pipeline from pretrained configuration.
        
        Args:
            model_name: Model variant ("mobile" or "server").
            det_providers: ONNX Runtime providers for detection.
            rec_providers: ONNX Runtime providers for recognition.
            visualize: Enable visualization.
            
        Returns:
            OCRPipeline instance.
        """
        config = OCRConfig.from_pretrained(model_name)
        return cls(config, det_providers, rec_providers, visualize)
    
    @classmethod
    def from_config(
        cls,
        config_path: Union[str, Path],
        det_providers: Optional[List[str]] = None,
        rec_providers: Optional[List[str]] = None,
        visualize: bool = False
    ) -> "OCRPipeline":
        """
        Create pipeline from YAML config file.
        
        Args:
            config_path: Path to YAML configuration file.
            det_providers: ONNX Runtime providers for detection.
            rec_providers: ONNX Runtime providers for recognition.
            visualize: Enable visualization.
            
        Returns:
            OCRPipeline instance.
        """
        config = OCRConfig.from_yaml(str(config_path))
        return cls(config, det_providers, rec_providers, visualize)
    
    @classmethod
    def from_model_paths(
        cls,
        det_model_path: str,
        rec_model_path: str,
        dict_path: Optional[str] = None,
        det_providers: Optional[List[str]] = None,
        rec_providers: Optional[List[str]] = None,
        visualize: bool = False,
        **kwargs
    ) -> "OCRPipeline":
        """
        Create pipeline directly from model paths.
        
        Args:
            det_model_path: Path to detection ONNX model.
            rec_model_path: Path to recognition ONNX model.
            dict_path: Path to character dictionary.
            det_providers: ONNX Runtime providers for detection.
            rec_providers: ONNX Runtime providers for recognition.
            visualize: Enable visualization.
            **kwargs: Additional config parameters.
            
        Returns:
            OCRPipeline instance.
        """
        if dict_path is None:
            package_dir = Path(__file__).parent
            dict_path = str(package_dir / "data/dict/ppocrv5_dict.txt")
        
        det_config = DetectorConfig(path=det_model_path, **kwargs)
        rec_config = RecognizerConfig(path=rec_model_path, dict_path=dict_path)
        config = OCRConfig(det=det_config, rec=rec_config)
        
        return cls(config, det_providers, rec_providers, visualize)

    def __call__(
        self,
        image: Union[str, np.ndarray, Image.Image],
        rec_batch_size: int = 64
    ) -> List[OCRResult]:
        """
        Run OCR on input image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image).
            rec_batch_size: Batch size for recognition.
            
        Returns:
            List of OCRResult objects.
        """
        # 1. Preprocess Input Image
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not read image from path: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        # 2. Detect text boxes
        det_time_start = time.time()
        dt_boxes = self.detector.detect(image)
        logger.info(f"Detection time: {time.time() - det_time_start:.3f}s")
        
        if not dt_boxes or len(dt_boxes[0]) == 0:
            return []
            
        points = dt_boxes[0][0]
        
        # 3. Crop detected regions
        cropped_imgs = []
        for pts in points:
            cropped_img = self.detector.crop_poly.get_minarea_rect_crop(image, pts)
            cropped_imgs.append(cropped_img)

        # 4. Recognize text in batches
        rec_results = []
        num_crops = len(cropped_imgs)
        
        rec_time_start = time.time()
        if num_crops > 0:
            for i in range(0, num_crops, rec_batch_size):
                batch_crops = cropped_imgs[i : i + rec_batch_size]
                batch_results = self.recognizer.recognize(batch_crops)
                rec_results.extend(batch_results)
        logger.info(f"Recognition time: {time.time() - rec_time_start:.3f}s")

        # 5. Pack results
        results = []
        for pts, (text, score) in zip(points, rec_results):
            result = OCRResult(point=pts, text=text, score=score)
            results.append(result)

        # 6. Optional visualization
        if self.visualizer is not None:
            vis_image = self.visualizer.draw(image, results)["ocr_res_img"]
            
            if isinstance(vis_image, np.ndarray):
                vis_image = Image.fromarray(vis_image)
            
            save_dir = Path(self.config.visualize.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            vis_image.save(save_dir / "ocr_result.png")
        
        return results