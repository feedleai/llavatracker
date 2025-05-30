import os
import requests
import json
import base64
from typing import Optional
import numpy as np
from PIL import Image
import io

from ..reid_types import BoundingBox, AppearanceDescription
from .base import BaseAppearanceExtractor

class LLaVAExtractor(BaseAppearanceExtractor):
    """LLaVA-based appearance description extractor using local Ollama server."""
    
    def __init__(self):
        self.server_url = None
        self.model_name = None
    
    def initialize(self, config: dict) -> None:
        """Initialize LLaVA local server settings."""
        self.server_url = config.get("server_url", "http://localhost:11434")
        self.model_name = config.get("model_name", "llava")
        
        # Test if Ollama server is running
        try:
            response = requests.get(f"{self.server_url}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"Warning: Ollama server not responding at {self.server_url}")
                print("Make sure to start Ollama with: ollama run llava")
        except requests.exceptions.RequestException:
            print(f"Warning: Cannot connect to Ollama server at {self.server_url}")
            print("Make sure to start Ollama with: ollama run llava")
    
    def extract(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        track_id: int
    ) -> Optional[AppearanceDescription]:
        """Extract detailed appearance description using local LLaVA model via Ollama."""
        if not self.server_url:
            return None
            
        # Crop and convert to PIL Image
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        pil_image = Image.fromarray(crop)
        
        # Convert to base64 for Ollama
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        # New structured prompt for consistent output
        reid_prompt = """You are a visual-attribute extractor for person re-identification.  
Inspect the image and output **only** the following JSON (no extra text):

{
  "gender": "male | female | unknown",
  "age": "child | teen | young_adult | middle_aged | elderly",
  "hair": {
      "color": "black | brown | blonde | red | gray | white | bald | unknown",
      "style": "short | medium | long | curly | straight | shaved | bald | unknown"
  },
  "top": {
      "type": "t-shirt | shirt | polo | sweater | jacket | hoodie | coat | dress | unknown",
      "color": "black | white | gray | red | orange | yellow | green | blue | purple | pink | brown | beige"
  },
  "bottom": {
      "type": "jeans | trousers | shorts | skirt | dress | leggings | unknown",
      "color": "same palette as above"
  },
  "shoes": {
      "type": "sneakers | dress_shoes | boots | sandals | heels | loafers | unknown",
      "color": "same palette as above"
  },
  "accessories": [
      "glasses","sunglasses","hat","cap","beanie",
      "bag","backpack","handbag","watch","bracelet","necklace",
      "scarf","earphones","mask"
  ],                      
  "dominant_outfit_colors": ["color1","color2","color3"]   
}

Rules:
- Use ONLY the listed values (lowercase, exact spelling).
- If uncertain, choose "unknown".
- Do **not** add or remove keys.
- Return valid JSON only."""
        
        # Call local Ollama server
        try:
            # For Ollama, we use the /api/generate endpoint
            response = requests.post(
                f"{self.server_url}/api/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "prompt": reid_prompt,
                    "images": [img_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0,  # Low temperature for consistent JSON output
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract the response text from Ollama format
            if "response" in data:
                response_text = data["response"]
                
                # Parse the JSON response
                parsed_data = self._parse_llava_response(response_text)
                return AppearanceDescription(**parsed_data)
            else:
                print(f"No response from LLaVA for track {track_id}")
                return None
            
        except requests.exceptions.RequestException as e:
            print(f"Network error with LLaVA server for track {track_id}: {e}")
            return None
        except Exception as e:
            print(f"LLaVA processing error for track {track_id}: {e}")
            return None
    
    def _parse_llava_response(self, response_text: str) -> dict:
        """Parse and validate LLaVA response data."""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                response_data = json.loads(json_str)
            else:
                # If no JSON found, return empty structure
                return self._get_empty_appearance()
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from LLaVA response: {response_text}")
            return self._get_empty_appearance()
        
        # Map LLaVA response to our AppearanceDescription structure
        parsed = {
            "gender_guess": response_data.get("gender", "unknown"),
            "age_range": response_data.get("age", "unknown"),
            
            # Hair information
            "hair_color": response_data.get("hair", {}).get("color", "unknown"),
            "hair_style": response_data.get("hair", {}).get("style", "unknown"),
            
            # Upper clothing
            "shirt_color": response_data.get("top", {}).get("color", "unknown"),
            "shirt_type": response_data.get("top", {}).get("type", "unknown"),
            
            # Lower clothing
            "pants_color": response_data.get("bottom", {}).get("color", "unknown"),
            "pants_type": response_data.get("bottom", {}).get("type", "unknown"),
            
            # Footwear
            "shoe_color": response_data.get("shoes", {}).get("color", "unknown"),
            "shoe_type": response_data.get("shoes", {}).get("type", "unknown"),
            
            # Additional features
            "accessories": response_data.get("accessories", []),
            "dominant_colors": response_data.get("dominant_outfit_colors", []),
            
            # Legacy fields for compatibility - create meaningful descriptions
            "hair": f"{response_data.get('hair', {}).get('color', 'unknown')} {response_data.get('hair', {}).get('style', '')}".strip(),
            "upper_clothing": f"{response_data.get('top', {}).get('color', 'unknown')} {response_data.get('top', {}).get('type', '')}".strip(),
            "lower_clothing": f"{response_data.get('bottom', {}).get('color', 'unknown')} {response_data.get('bottom', {}).get('type', '')}".strip(),
            "footwear": f"{response_data.get('shoes', {}).get('color', 'unknown')} {response_data.get('shoes', {}).get('type', '')}".strip()
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
        """No cleanup needed for local server extractor."""
        pass