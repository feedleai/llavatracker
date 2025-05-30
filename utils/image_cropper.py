"""
Utility for cropping images based on bounding boxes.
"""

import numpy as np
from typing import Tuple, Optional
from ..types import BoundingBox

class ImageCropper:
    """Utility class for cropping images with padding and safety checks."""
    
    @staticmethod
    def crop(
        image: np.ndarray,
        bbox: BoundingBox,
        padding: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        Crop image with optional padding.
        
        Args:
            image: Input image (H, W, C)
            bbox: Bounding box (x1, y1, x2, y2)
            padding: Padding factor (0.0 to 1.0)
            
        Returns:
            Cropped image or None if invalid
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Add padding
        if padding > 0:
            width = x2 - x1
            height = y2 - y1
            x1 = max(0, x1 - width * padding)
            y1 = max(0, y1 - height * padding)
            x2 = min(w, x2 + width * padding)
            y2 = min(h, y2 + height * padding)
        
        # Ensure valid coordinates
        x1, y1 = map(int, (max(0, x1), max(0, y1)))
        x2, y2 = map(int, (min(w, x2), min(h, y2)))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return image[y1:y2, x1:x2] 