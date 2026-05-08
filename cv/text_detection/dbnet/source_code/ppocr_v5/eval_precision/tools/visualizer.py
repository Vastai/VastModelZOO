import traceback
import random
import math
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
from ..schema import OCRResult
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger


class FontRenderer:
    """Handles font creation and text rendering for OCR visualization."""
    
    def __init__(self, font_path: str):
        """
        Initialize FontRenderer with a font file path.
        
        Args:
            font_path: Path to the TrueType font file.
        """
        self.font_path = font_path
        self._validate_font_path()
    
    def _validate_font_path(self) -> None:
        """Validate that the font file exists."""
        if self.font_path and not Path(self.font_path).exists():
            logger.warning(f"Font file not found: {self.font_path}")
    
    def create_horizontal_font(
        self, 
        text: str, 
        box_size: Tuple[int, int]
    ) -> ImageFont.FreeTypeFont:
        """
        Create a font sized to fit horizontal text within the given box.
        
        Args:
            text: The text to render.
            box_size: Tuple of (width, height) for the bounding box.
            
        Returns:
            A PIL ImageFont object sized appropriately.
        """
        width, height = box_size
        font_size = int(height * 0.8)
        font_size = max(font_size, 10)
        
        font = ImageFont.truetype(self.font_path, font_size, encoding="utf-8")
        text_length = font.getlength(text)
        
        # Scale down if text is too wide
        if text_length > width:
            font_size = int(font_size * width / text_length)
            font_size = max(font_size, 10)
            font = ImageFont.truetype(self.font_path, font_size, encoding="utf-8")
        
        return font
    
    def create_vertical_font(
        self, 
        text: str, 
        box_size: Tuple[int, int], 
        scale: float = 1.2
    ) -> ImageFont.FreeTypeFont:
        """
        Create a font sized to fit vertical text within the given box.
        
        Args:
            text: The text to render.
            box_size: Tuple of (width, height) for the bounding box.
            scale: Scaling factor for font size adjustment.
            
        Returns:
            A PIL ImageFont object sized appropriately.
        """
        width, height = box_size
        char_count = len(text) if text else 1
        
        # Calculate base font size based on height and character count
        base_font_size = int(height / char_count * 0.8 * scale)
        base_font_size = max(base_font_size, 10)
        
        font = ImageFont.truetype(self.font_path, base_font_size, encoding="utf-8")
        
        # Check if any character exceeds the box width
        max_char_width = max([font.getlength(c) for c in text]) if text else 0
        
        if max_char_width > width:
            new_size = int(base_font_size * width / max_char_width)
            new_size = max(new_size, 10)
            font = ImageFont.truetype(self.font_path, new_size, encoding="utf-8")
        
        return font
    
    def draw_vertical_text(
        self,
        draw: ImageDraw.ImageDraw,
        position: Tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: Tuple[int, int, int] = (0, 0, 0),
        line_spacing: int = 2
    ) -> None:
        """
        Draw text vertically (one character per line).
        
        Args:
            draw: PIL ImageDraw object.
            position: Starting (x, y) position.
            text: Text to draw.
            font: Font to use.
            fill: RGB color tuple.
            line_spacing: Pixels between characters.
        """
        x, y = position
        for char in text:
            draw.text((x, y), char, font=font, fill=fill)
            bbox = font.getbbox(char)
            char_height = bbox[3] - bbox[1]
            y += char_height + line_spacing


class BoxProcessor:
    """Handles bounding box operations and transformations."""
    
    @staticmethod
    def get_minimum_area_rect(points: np.ndarray) -> np.ndarray:
        """
        Get the minimum area rectangle for the given points.
        
        Args:
            points: An array of 2D points.
            
        Returns:
            Array of 4 corner points ordered clockwise from top-left.
        """
        bounding_box = cv2.minAreaRect(points)
        box_points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        
        # Determine point ordering based on y-coordinates
        if box_points[1][1] > box_points[0][1]:
            idx_a, idx_d = 0, 1
        else:
            idx_a, idx_d = 1, 0
            
        if box_points[3][1] > box_points[2][1]:
            idx_b, idx_c = 2, 3
        else:
            idx_b, idx_c = 3, 2
        
        ordered_box = np.array([
            box_points[idx_a],
            box_points[idx_b],
            box_points[idx_c],
            box_points[idx_d]
        ]).astype(np.int32)
        
        return ordered_box
    
    @staticmethod
    def calculate_box_dimensions(box: np.ndarray) -> Tuple[int, int]:
        """
        Calculate width and height of a quadrilateral box.
        
        Args:
            box: 4x2 array of corner points.
            
        Returns:
            Tuple of (width, height).
        """
        height = int(math.sqrt(
            (box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2
        ))
        width = int(math.sqrt(
            (box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2
        ))
        return width, height
    
    @staticmethod
    def is_vertical_text(box: np.ndarray, threshold_ratio: float = 2.0) -> bool:
        """
        Determine if the box contains vertical text based on aspect ratio.
        
        Args:
            box: 4x2 array of corner points.
            threshold_ratio: Height/width ratio threshold for vertical detection.
            
        Returns:
            True if text should be rendered vertically.
        """
        width, height = BoxProcessor.calculate_box_dimensions(box)
        return height > threshold_ratio * width and height > 30


class TextBoxRenderer:
    """Renders text within transformed bounding boxes."""
    
    def __init__(self, font_renderer: FontRenderer):
        """
        Initialize TextBoxRenderer.
        
        Args:
            font_renderer: FontRenderer instance for text rendering.
        """
        self.font_renderer = font_renderer
        self.box_processor = BoxProcessor()
    
    def render_text_in_box(
        self,
        image_size: Tuple[int, int],
        box: np.ndarray,
        text: str
    ) -> np.ndarray:
        """
        Render text inside a transformed bounding box.
        
        Args:
            image_size: Output image size (width, height).
            box: 4x2 array defining box corners.
            text: Text to render.
            
        Returns:
            Image with text rendered and transformed to fit the box.
        """
        box_width, box_height = self.box_processor.calculate_box_dimensions(box)
        box_width = max(box_width, 1)
        box_height = max(box_height, 1)
        
        # Create text image
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        draw = ImageDraw.Draw(img_text)
        
        if text:
            if self.box_processor.is_vertical_text(box):
                font = self.font_renderer.create_vertical_font(
                    text, (box_width, box_height)
                )
                self.font_renderer.draw_vertical_text(
                    draw, (0, 0), text, font, fill=(0, 0, 0)
                )
            else:
                font = self.font_renderer.create_horizontal_font(
                    text, (box_width, box_height)
                )
                draw.text((0, 0), text, fill=(0, 0, 0), font=font)
        
        # Apply perspective transform
        src_points = np.float32([
            [0, 0],
            [box_width, 0],
            [box_width, box_height],
            [0, box_height]
        ])
        dst_points = np.array(box, dtype=np.float32)
        
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        img_text_array = np.array(img_text, dtype=np.uint8)
        warped_image = cv2.warpPerspective(
            img_text_array,
            transform_matrix,
            image_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return warped_image


class OCRVisualizer:
    """
    Main visualizer class for OCR detection and recognition results.
    
    This class provides functionality to draw bounding boxes and recognized
    text on images for visualization purposes.
    """
    
    def __init__(
        self,
        font_path: str = "",
        box_thickness: int = 2,
        random_seed: int = 0
    ):
        """
        Initialize OCRVisualizer.
        
        Args:
            font_path: Path to TrueType font file for text rendering.
            box_thickness: Line thickness for drawing bounding boxes.
            random_seed: Seed for random color generation (for reproducibility).
        """
        self.font_path = font_path
        self.box_thickness = box_thickness
        self.random_seed = random_seed
        
        self.font_renderer = FontRenderer(font_path) if font_path else None
        self.text_box_renderer = TextBoxRenderer(self.font_renderer) if self.font_renderer else None
        self.box_processor = BoxProcessor()
    
    def _generate_random_color(self) -> Tuple[int, int, int]:
        """Generate a random RGB color."""
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    
    def _process_polygon_box(
        self,
        box: np.ndarray,
        draw: ImageDraw.ImageDraw,
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Process and draw a polygon box with more than 4 points.
        
        Args:
            box: Array of polygon points.
            draw: PIL ImageDraw object.
            color: RGB color tuple.
            
        Returns:
            Simplified 4-point bounding box.
        """
        pts = [(x, y) for x, y in box.tolist()]
        draw.polygon(pts, outline=color, width=8)
        
        # Convert to minimum area rectangle
        box = self.box_processor.get_minimum_area_rect(box)
        
        # Adjust box for text rendering
        height = int(0.5 * (max(box[:, 1]) - min(box[:, 1])))
        box[:2, 1] = np.mean(box[:, 1])
        box[2:, 1] = np.mean(box[:, 1]) + min(20, height)
        
        return box
    
    def _draw_detection_overlay(
        self,
        image: Image.Image,
        boxes: List[np.ndarray],
        colors: List[Tuple[int, int, int]]
    ) -> Image.Image:
        """
        Draw detection boxes as colored overlays on the image.
        
        Args:
            image: PIL Image to draw on.
            boxes: List of bounding boxes.
            colors: List of colors corresponding to each box.
            
        Returns:
            Image with detection overlays.
        """
        draw = ImageDraw.Draw(image)
        
        for box, color in zip(boxes, colors):
            box = np.array(box)
            
            if len(box) > 4:
                box = self._process_polygon_box(box, draw, color)
            
            box_pts = [(int(x), int(y)) for x, y in box.tolist()]
            draw.polygon(box_pts, fill=color)
        
        return image
    
    def _create_text_panel(
        self,
        image_size: Tuple[int, int],
        boxes: List[np.ndarray],
        texts: List[str],
        colors: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """
        Create a panel with recognized text rendered in corresponding boxes.
        
        Args:
            image_size: Size of the output panel (width, height).
            boxes: List of bounding boxes.
            texts: List of recognized texts.
            colors: List of colors for box outlines.
            
        Returns:
            Numpy array representing the text panel image.
        """
        width, height = image_size
        text_panel = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        if self.text_box_renderer is None:
            return text_panel
        
        for box, text, color in zip(boxes, texts, colors):
            try:
                box = np.array(box)
                
                if len(box) > 4:
                    box = self.box_processor.get_minimum_area_rect(box)
                    box_height = int(0.5 * (max(box[:, 1]) - min(box[:, 1])))
                    box[:2, 1] = np.mean(box[:, 1])
                    box[2:, 1] = np.mean(box[:, 1]) + min(20, box_height)
                
                # Render text in box
                text_image = self.text_box_renderer.render_text_in_box(
                    (width, height), box, text
                )
                
                # Draw box outline
                pts = np.array(box, np.int32).reshape((-1, 1, 2))
                cv2.polylines(text_image, [pts], True, color, self.box_thickness)
                
                # Combine with panel
                text_panel = cv2.bitwise_and(text_panel, text_image)
                
            except Exception as e:
                logger.error(f"Error rendering text box: {e}")
                logger.debug(traceback.format_exc())
                continue
        
        return text_panel
    
    def draw(
        self,
        image: np.ndarray,
        ocr_results: List[OCRResult],
        detection_only: bool = False,
        layout: str = "horizontal"
    ) -> Dict[str, Image.Image]:
        """
        Draw OCR results on an image.
        
        Args:
            image: Input image as numpy array (BGR format).
            boxes: List of bounding boxes.
            texts: List of recognized texts (optional if detection_only=True).
            detection_only: If True, only draw detection boxes without text panel.
            layout: Layout mode - "horizontal" or "vertical" for side-by-side display.
            
        Returns:
            Dictionary containing result images.
        """
        texts = [res.text for res in ocr_results]
        boxes = [res.point for res in ocr_results]
        if texts is None:
            texts = [""] * len(boxes)
        
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize random seed for reproducible colors
        random.seed(self.random_seed)
        colors = [self._generate_random_color() for _ in boxes]
        
        # Create detection overlay
        img_detection = Image.fromarray(image_rgb.copy())
        img_detection = self._draw_detection_overlay(img_detection, boxes, colors)
        
        # Blend with original image
        img_blended = Image.blend(
            Image.fromarray(image_rgb),
            img_detection,
            alpha=0.5
        )
        
        if detection_only:
            return {"ocr_res_img": img_blended}
        
        # Create text panel
        random.seed(self.random_seed)  # Reset seed for consistent colors
        colors = [self._generate_random_color() for _ in boxes]
        text_panel = self._create_text_panel((width, height), boxes, texts, colors)
        
        # Combine detection and text panels
        if layout == "horizontal":
            result_image = Image.new("RGB", (width * 2, height), (255, 255, 255))
            result_image.paste(img_blended, (0, 0))
            result_image.paste(Image.fromarray(text_panel), (width, 0))
        else:  # vertical layout
            result_image = Image.new("RGB", (width, height * 2), (255, 255, 255))
            result_image.paste(img_blended, (0, 0))
            result_image.paste(Image.fromarray(text_panel), (0, height))
        
        return {"ocr_res_img": result_image}
    
    def __call__(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray],
        texts: Optional[List[str]] = None,
        detection_only: bool = False
    ) -> Dict[str, Image.Image]:
        """
        Callable interface for drawing OCR results.
        
        Args:
            image: Input image as numpy array (BGR format).
            boxes: List of bounding boxes.
            texts: List of recognized texts.
            detection_only: If True, only draw detection boxes.
            
        Returns:
            Dictionary containing result images.
        """
        return self.draw(image, boxes, texts, detection_only)


# Convenience function for backward compatibility
def draw_image(
    image: np.ndarray,
    boxes: List[np.ndarray],
    txts: List[str],
    font_path: str = "",
    box_thickness: int = 2,
    det_only: bool = False
) -> Dict[str, Image.Image]:
    """
    Draw OCR results on an image (backward compatible function).
    
    Args:
        image: Input image as numpy array (BGR format).
        boxes: List of bounding boxes.
        txts: List of recognized texts.
        font_path: Path to font file.
        box_thickness: Thickness of box outlines.
        det_only: If True, only draw detection boxes.
        
    Returns:
        Dictionary containing result images.
    """
    visualizer = OCRVisualizer(
        font_path=font_path,
        box_thickness=box_thickness
    )
    return visualizer.draw(image, boxes, txts, detection_only=det_only)