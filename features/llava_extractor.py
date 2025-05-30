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
    """LLaVA-based appearance description extractor for detailed color and style analysis."""
    
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
        """Extract detailed appearance description using LLaVA API."""
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
        
        # Enhanced prompt for specific color and style extraction
        detailed_prompt = """
        Analyze this person's appearance and provide detailed information in the following JSON format:
        {
            "gender_guess": "male/female/unknown",
            "age_range": "child/teenager/young_adult/middle_aged/elderly",
            "hair_color": "specific color (e.g., black, brown, blonde, gray, red, etc.)",
            "hair_style": "short/long/curly/straight/bald/etc.",
            "shirt_color": "specific color of upper clothing",
            "shirt_type": "t-shirt/button-down/polo/sweater/jacket/hoodie/tank-top/etc.",
            "pants_color": "specific color of lower clothing",
            "pants_type": "jeans/dress-pants/shorts/skirt/dress/leggings/etc.",
            "shoe_color": "specific color of footwear",
            "shoe_type": "sneakers/dress-shoes/boots/sandals/heels/etc.",
            "accessories": ["list of visible accessories like glasses, hat, bag, watch, etc."],
            "dominant_colors": ["top 3 most prominent colors in outfit"]
        }
        
        Be specific about colors (e.g., "dark blue" instead of just "blue"). If you can't see something clearly, use "unknown".
        """
        
        # Call LLaVA API
        try:
            response = requests.post(
                self.api_endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"image": img_byte_arr},
                json={
                    "prompt": detailed_prompt,
                    "format": "json",
                    "max_tokens": 500
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse response into AppearanceDescription
            # Handle potential API response variations
            if "response" in data:
                appearance_data = data["response"]
            else:
                appearance_data = data
                
            # Ensure we have the expected structure
            parsed_data = self._parse_llava_response(appearance_data)
            return AppearanceDescription(**parsed_data)
            
        except Exception as e:
            print(f"LLaVA API error for track {track_id}: {e}")
            return None
    
    def _parse_llava_response(self, response_data) -> dict:
        """Parse and validate LLaVA response data."""
        if isinstance(response_data, str):
            try:
                response_data = json.loads(response_data)
            except json.JSONDecodeError:
                # If JSON parsing fails, return empty structure
                return self._get_empty_appearance()
        
        # Map LLaVA response to our AppearanceDescription structure
        parsed = {
            "gender_guess": response_data.get("gender_guess", "unknown"),
            "age_range": response_data.get("age_range", "unknown"),
            
            # Hair information
            "hair_color": response_data.get("hair_color", "unknown"),
            "hair_style": response_data.get("hair_style", "unknown"),
            
            # Upper clothing
            "shirt_color": response_data.get("shirt_color", "unknown"),
            "shirt_type": response_data.get("shirt_type", "unknown"),
            
            # Lower clothing
            "pants_color": response_data.get("pants_color", "unknown"),
            "pants_type": response_data.get("pants_type", "unknown"),
            
            # Footwear
            "shoe_color": response_data.get("shoe_color", "unknown"),
            "shoe_type": response_data.get("shoe_type", "unknown"),
            
            # Additional features
            "accessories": response_data.get("accessories", []),
            "dominant_colors": response_data.get("dominant_colors", []),
            
            # Legacy fields for compatibility
            "hair": f"{response_data.get('hair_color', 'unknown')} {response_data.get('hair_style', '')}".strip(),
            "upper_clothing": f"{response_data.get('shirt_color', 'unknown')} {response_data.get('shirt_type', '')}".strip(),
            "lower_clothing": f"{response_data.get('pants_color', 'unknown')} {response_data.get('pants_type', '')}".strip(),
            "footwear": f"{response_data.get('shoe_color', 'unknown')} {response_data.get('shoe_type', '')}".strip()
        }
        
        return parsed
    
    def _get_empty_appearance(self) -> dict:
        """Return empty appearance structure."""
        return {
            "gender_guess": "unknown",
            "age_range": "unknown",
            "hair_color": "unknown",
            "hair_style": "unknown",
            "shirt_color": "unknown",
            "shirt_type": "unknown",
            "pants_color": "unknown",
            "pants_type": "unknown",
            "shoe_color": "unknown",
            "shoe_type": "unknown",
            "accessories": [],
            "dominant_colors": [],
            "hair": "unknown",
            "upper_clothing": "unknown",
            "lower_clothing": "unknown",
            "footwear": "unknown"
        }
    
    def cleanup(self) -> None:
        """No cleanup needed for API-based extractor."""
        pass