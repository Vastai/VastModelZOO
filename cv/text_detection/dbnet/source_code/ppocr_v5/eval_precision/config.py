from typing import Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
import yaml
import os
import zipfile


@dataclass
class DetectorConfig:
    """Configuration for text detection model."""
    path: str = ''
    resize_long: int = 960
    input_shape: Optional[list] = field(default_factory=lambda: [3, 960, 960])
    use_fixed_shape: bool = False
    thresh: float = 0.3
    box_thresh: float = 0.6
    unclip_ratio: float = 1.5
    max_candidates: int = 1000

    # vacc config
    vacc_path: str = ''
    vdsp_path: str = ''
    device_id: int = 0
    batch: int = 1

@dataclass
class RecognizerConfig:
    """Configuration for text recognition model."""
    path: str = ''
    input_shape: list = field(default_factory=lambda: [3, 48, 320])
    use_fixed_shape: bool = False
    use_letterbox: bool = True
    dict_path: str = "ppocrv5_onnx/data/dict/ppocrv5_dict.txt"

    # vacc config
    vacc_path: str = ''
    vdsp_path: str = ''
    device_id: int = 0
    batch: int = 1
    
@dataclass
class VisualizeConfig:
    """Configuration for visualization."""
    font_path: Optional[str] = "ppocrv5_onnx/data/fonts/simfang.ttf"
    save_dir: str = "output"
    box_thickness: int = 2

@dataclass
class OCRConfig:
    """Main OCR configuration."""
    det: DetectorConfig
    rec: RecognizerConfig
    visualize: VisualizeConfig = field(default_factory=VisualizeConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OCRConfig":
        """Create config from dictionary."""
        engine = config_dict.get("engine", {})
        model = engine.get("model", {})
        
        det_config = DetectorConfig(**model.get("det", {}))
        rec_config = RecognizerConfig(**model.get("rec", {}))
        vis_config = VisualizeConfig(**config_dict.get("visualize", {}))
        
        return cls(det=det_config, rec=rec_config, visualize=vis_config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OCRConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_pretrained(cls, model_name: str = "mobile") -> "OCRConfig":
        """Load predefined configuration."""
        import urllib.request
        
        package_dir = Path(__file__).parent
        
        # Define base URLs for downloading models (adjust as needed)
        # base_url = "https://paddleocr.bj.bcebos.com/PP-OCRv5/"
        
        def download_if_not_exists(file_path: Path, url: str):
            if not file_path.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {file_path} from {url}")
            if url.endswith('.zip'):
                temp_zip = file_path.parent / "temp.zip"
                urllib.request.urlretrieve(url, temp_zip)
                with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                    zip_ref.extractall(file_path)
                temp_zip.unlink()
            else:
                urllib.request.urlretrieve(url, file_path)
        
        if model_name == "mobile":
            
            model_path = package_dir / "models"
            dict_path = package_dir / "data/dict/ppocrv5_dict.txt"
            font_path = package_dir / "data/fonts/simfang.ttf"
            det_path = package_dir / "models/PP-OCRv5_mobile_det/inference.onnx"
            rec_path = package_dir / "models/PP-OCRv5_mobile_rec/inference.onnx"

            if not det_path.exists():
                download_if_not_exists(model_path, 
                                       url="https://github.com/HoVDuc/ppocrv5-onnx/releases/download/v1.0.0/PP-OCRv5_mobile_det.zip")
            if not rec_path.exists():
                download_if_not_exists(model_path, 
                                       url="https://github.com/HoVDuc/ppocrv5-onnx/releases/download/v1.0.0/PP-OCRv5_mobile_rec.zip")

            return cls(
                det=DetectorConfig(path=str(det_path)),
                rec=RecognizerConfig(
                    path=str(rec_path),
                    dict_path=str(dict_path)
                ),
                visualize=VisualizeConfig(font_path=str(font_path))
            )
        elif model_name == "server":
            base_url = ""
            det_path = package_dir / "models/PP-OCRv5_server_det/inference.onnx"
            rec_path = package_dir / "models/PP-OCRv5_server_rec/inference.onnx"
            dict_path = package_dir / "data/dict/ppocrv5_dict.txt"
            
            download_if_not_exists(det_path, f"{base_url}server/det/inference.onnx")
            download_if_not_exists(rec_path, f"{base_url}server/rec/inference.onnx")
            download_if_not_exists(dict_path, f"{base_url}dict/ppocrv5_dict.txt")
            
            return cls(
                det=DetectorConfig(path=str(det_path), resize_long=1280),
                rec=RecognizerConfig(
                    path=str(rec_path),
                    dict_path=str(dict_path)
                )
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)