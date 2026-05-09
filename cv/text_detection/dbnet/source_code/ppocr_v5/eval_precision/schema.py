import numpy as np
from dataclasses import dataclass
from typing import List, Union, Tuple

@dataclass
class OCRResult:
    text: str
    score: float
    point: Union[List[int], np.ndarray]

    def xyxy(self) -> Tuple[int, int, int, int]:
        """
        Convert quadrilateral points to xyxy bounding box format.
        
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        if isinstance(self.point, list):
            point = np.array(self.point)
        else:
            point = self.points
            
        x_coords = point[:, 0]
        y_coords = point[:, 1]
        
        x_min = int(np.min(x_coords))
        y_min = int(np.min(y_coords))
        x_max = int(np.max(x_coords))
        y_max = int(np.max(y_coords))
        
        return (x_min, y_min, x_max, y_max)

    def to_dict(self):
        return {
            "text": self.text,
            "score": self.score,
            "box": self.box.tolist() if isinstance(self.box, np.ndarray) else self.box
        }