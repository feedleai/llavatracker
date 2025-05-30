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
    """LLaVA-based appearance description extractor using local vLLM server."""
    
    def __init__(self):
        self.server_url = None
        self.model_name = None
    
    def initialize(self, config: dict) -> None:
        """Initialize LLaVA local server settings."""
        self.server_url = config.get("server_url", "http://localhost:8000")
        self.model_name = config.get("model_name", "liuhaotian/llava-v1.5-7b")
        
        # Test if server is running
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code != 200:
                print(f"Warning: vLLM server not responding at {self.server_url}")
                print("Make sure to start the server with: vllm serve liuhaotian/llava-v1.5-7b")
        except requests.exceptions.RequestException:
            print(f"Warning: Cannot connect to vLLM server at {self.server_url}")
            print("Make sure to start the server with: vllm serve liuhaotian/llava-v1.5-7b")
    
    def extract(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        track_id: int
    ) -> Optional[AppearanceDescription]:
        """Extract detailed appearance description using local LLaVA model."""
        if not self.server_url:
            return None
            
        # Crop and convert to PIL Image
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        pil_image = Image.fromarray(crop)
        
        # Convert to base64 for vLLM
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        # Enhanced prompt for specific color and style extraction
        detailed_prompt = """Analyze this person's appearance and provide detailed information in the following JSON format:
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

Be specific about colors (e.g., "dark blue" instead of just "blue"). If you can't see something clearly, use "unknown". Only return the JSON, no other text."""
        
        # Call local vLLM server
        try:
            # For vLLM with vision models, we need to use the chat completions endpoint
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": detailed_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 500,
                    "temperature": 0.1,  # Low temperature for consistent JSON output
                    "top_p": 0.9
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract the response text
            if "choices" in data and len(data["choices"]) > 0:
                response_text = data["choices"][0]["message"]["content"]
                
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
        """No cleanup needed for local server extractor."""
        pass