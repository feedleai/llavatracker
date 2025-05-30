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
        reid_prompt = """Analyze the provided image of a person with a focus on attributes crucial for re-identification. Provide detailed information in the following JSON format:
{
    "gender_guess": "male/female/unknown",
    "age_range": "child/teenager/young_adult/middle_aged/elderly",
    "hair_color": "specific color (e.g., black, brown, blonde, gray, red, salt-and-pepper). If covered or not visible, use 'unknown' or 'covered'.",
    "hair_style": "short/long/curly/straight/wavy/bald/ponytail/bun/braids/afro/covered_by_headwear/etc. Be specific if possible (e.g., 'short and spiky', 'long and straight').",
    "headwear_type": "none/cap/beanie/hat (specify type e.g., fedora, sunhat)/hood_up/headband/scarf_on_head/helmet/etc.",
    "headwear_color": "specific color of headwear (e.g., 'bright red', 'dark gray'). If multiple colors, list prominent ones. Use 'none' if no headwear.",
    "facial_features_accessories": ["list any visible facial accessories like 'glasses (specify frame color/style if possible, e.g., 'black thick-rimmed glasses')', 'sunglasses (specify lens/frame color)', 'face_mask (specify color)', 'earrings (specify color/type if clear)', 'piercings (location if clear)'. Use empty list [] if none."],
    "upper_clothing_color_primary": "specific primary color of the main upper body garment (e.g., 'navy blue', 'lime green', 'maroon').",
    "upper_clothing_color_secondary": ["list other distinct colors if present on the main upper garment (e.g., 'white stripes', 'yellow logo'). Use empty list [] if none."],
    "upper_clothing_type": "t-shirt/polo_shirt/button-down_shirt/blouse/sweater/sweatshirt/hoodie/jacket (specify type if possible e.g., 'denim jacket', 'leather jacket', 'windbreaker')/vest/tank_top/dress/etc.",
    "upper_clothing_pattern_or_print": "describe any pattern (e.g., 'horizontal_stripes', 'vertical_stripes', 'plaid', 'checked', 'floral', 'polka_dots', 'camouflage', 'abstract') or significant graphic/logo/text (e.g., 'Nike swoosh on left chest', 'band name text', 'large eagle graphic'). Use 'none' if plain.",
    "sleeve_length": "short_sleeve/long_sleeve/three_quarter_sleeve/sleeveless/rolled_up_sleeves/unknown.",
    "lower_clothing_color": "specific primary color of lower body clothing (e.g., 'dark wash blue' for jeans, 'khaki', 'black'). If 'dress' is upper_clothing_type, specify if lower part is distinct or 'same_as_upper'.",
    "lower_clothing_type": "jeans/dress_pants/casual_pants (e.g. chinos, khakis)/shorts/skirt (specify length e.g. mini, knee-length, maxi)/leggings/sweatpants/track_pants/cargo_pants/etc. If 'dress' is upper_clothing_type, use 'dress'.",
    "lower_clothing_pattern": "describe any pattern (e.g., 'pinstripes', 'camouflage', 'acid_wash'). Use 'none' if plain.",
    "footwear_color": "specific color(s) of footwear (e.g., 'white_with_red_accents', 'all_black', 'brown').",
    "footwear_type": "sneakers/trainers/athletic_shoes/dress_shoes (e.g. oxfords, loafers)/boots (specify type if possible e.g. ankle, combat, knee-high)/sandals/flip-flops/heels/flats/etc.",
    "carried_items_or_prominent_accessories": ["list and describe items being carried or very prominent non-clothing accessories (e.g., 'black_backpack', 'brown_leather_shoulder_bag', 'holding_red_umbrella', 'white_shopping_bag_with_logo', 'silver_watch_on_left_wrist', 'multiple_bracelets'). Use empty list [] if none."],
    "dominant_colors_overall_outfit": ["list the top 3-5 most prominent colors visible in the entire outfit, considering surface area and visual impact (e.g., 'navy_blue', 'white', 'khaki', 'red')."],
    "other_distinctive_visual_cues": "any other highly distinctive visual feature useful for re-identification not captured above (e.g., 'large_colorful_tattoo_on_right_forearm', 'person_is_using_crutches', 'bright_pink_hair_streak', 'wearing_a_name_badge'). Use 'none' if nothing else stands out."
}

Instructions for the model:
- Prioritize accuracy and detail for visual attributes.
- Be very specific about colors (e.g., "light blue" or "dark red" instead of just "blue" or "red"). Include metallic colors like "silver" or "gold" if applicable.
- If an attribute is occluded, ambiguous, or truly not determinable from the image, use "unknown". For list fields where nothing applies, use an empty list []. For string fields where "none" is an appropriate description (like patterns or headwear), use "none".
- Only return the populated JSON object. No other text before or after the JSON."""
        
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
        
        # Helper function to safely get string values (convert lists to appropriate strings)
        def safe_get_string(data, key, default="unknown"):
            value = data.get(key, default)
            if isinstance(value, list):
                if len(value) == 0:
                    return "none" if key in ["headwear_type", "upper_clothing_pattern_or_print", "lower_clothing_pattern", "other_distinctive_visual_cues"] else "unknown"
                else:
                    return str(value[0])  # Take first item if list is not empty
            return str(value) if value is not None else default
        
        # Helper function to safely get list values (convert strings to lists if needed)
        def safe_get_list(data, key, default=None):
            if default is None:
                default = []
            value = data.get(key, default)
            if isinstance(value, str):
                if value in ["none", "unknown", ""]:
                    return []
                else:
                    return [value]  # Convert single string to list
            elif isinstance(value, list):
                return value
            else:
                return default
        
        # Map LLaVA response to our AppearanceDescription structure with type safety
        parsed = {
            "gender_guess": safe_get_string(response_data, "gender_guess", "unknown"),
            "age_range": safe_get_string(response_data, "age_range", "unknown"),
            "hair_color": safe_get_string(response_data, "hair_color", "unknown"),
            "hair_style": safe_get_string(response_data, "hair_style", "unknown"),
            "headwear_type": safe_get_string(response_data, "headwear_type", "none"),
            "headwear_color": safe_get_string(response_data, "headwear_color", "none"),
            "facial_features_accessories": safe_get_list(response_data, "facial_features_accessories", []),
            "upper_clothing_color_primary": safe_get_string(response_data, "upper_clothing_color_primary", "unknown"),
            "upper_clothing_color_secondary": safe_get_list(response_data, "upper_clothing_color_secondary", []),
            "upper_clothing_type": safe_get_string(response_data, "upper_clothing_type", "unknown"),
            "upper_clothing_pattern_or_print": safe_get_string(response_data, "upper_clothing_pattern_or_print", "none"),
            "sleeve_length": safe_get_string(response_data, "sleeve_length", "unknown"),
            "lower_clothing_color": safe_get_string(response_data, "lower_clothing_color", "unknown"),
            "lower_clothing_type": safe_get_string(response_data, "lower_clothing_type", "unknown"),
            "lower_clothing_pattern": safe_get_string(response_data, "lower_clothing_pattern", "none"),
            "footwear_color": safe_get_string(response_data, "footwear_color", "unknown"),
            "footwear_type": safe_get_string(response_data, "footwear_type", "unknown"),
            "carried_items_or_prominent_accessories": safe_get_list(response_data, "carried_items_or_prominent_accessories", []),
            "dominant_colors_overall_outfit": safe_get_list(response_data, "dominant_colors_overall_outfit", []),
            "other_distinctive_visual_cues": safe_get_string(response_data, "other_distinctive_visual_cues", "none")
        }
        
        return parsed
    
    def _get_empty_appearance(self) -> dict:
        """Return empty appearance structure."""
        return {
            "gender_guess": "unknown",
            "age_range": "unknown",
            "hair_color": "unknown",
            "hair_style": "unknown",
            "headwear_type": "unknown",
            "headwear_color": "unknown",
            "facial_features_accessories": [],
            "upper_clothing_color_primary": "unknown",
            "upper_clothing_color_secondary": [],
            "upper_clothing_type": "unknown",
            "upper_clothing_pattern_or_print": "none",
            "sleeve_length": "unknown",
            "lower_clothing_color": "unknown",
            "lower_clothing_type": "unknown",
            "lower_clothing_pattern": "none",
            "footwear_color": "unknown",
            "footwear_type": "unknown",
            "carried_items_or_prominent_accessories": [],
            "dominant_colors_overall_outfit": [],
            "other_distinctive_visual_cues": "none"
        }
    
    def cleanup(self) -> None:
        """No cleanup needed for local server extractor."""
        pass