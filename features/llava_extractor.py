
import os
import requests
import json
from typing import Optional
import numpy as np
from PIL import Image
import io

from ..types import BoundingBox, AppearanceDescription
from .base import BaseAppearanceExtractor

class LLaVAExtractor(BaseAppearanceExtractor):
    """LLaVA-based appearance description extractor."""
    
    def __init__(self):
        self.api_key = os.getenv("LLAVA_API_KEY")
        self.api_endpoint = None
    
    def initialize(self, config: dict) -> None:
        """Initialize LLaVA API settings."""
        self.api_endpoint = config.get("api_endpoint")
        if not self.api_key:
            raise ValueError("LLAVA_API_KEY environment variable not set")
    
    def extract(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        track_id: int
    ) -> Optional[AppearanceDescription]:
        """Extract appearance description using LLaVA API."""
        if not self.api_endpoint or not self.api_key:
            return None
            
        # Crop and convert to PIL Image
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        pil_image = Image.fromarray(crop)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Call LLaVA API
        try:
            response = requests.post(
                self.api_endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"image": img_byte_arr},
                json={
                    "prompt": "Describe this person's appearance in detail, including gender, age range, hair, clothing, and accessories.",
                    "format": "json"
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse response into AppearanceDescription
            return AppearanceDescription(**data)
            
        except Exception as e:
            print(f"LLaVA API error for track {track_id}: {e}")
            return None
    
    def cleanup(self) -> None:
        """No cleanup needed for API-based extractor."""
        pass