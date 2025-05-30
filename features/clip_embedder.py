"""
CLIP feature extractor for person re-identification.
"""

import torch
import open_clip
from typing import Optional
import numpy as np

from ..types import BoundingBox
from .base import BaseFeatureExtractor

class CLIPEmbedder(BaseFeatureExtractor):
    """CLIP feature extractor using ViT-B/32 model."""
    
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def initialize(self, config: dict) -> None:
        """Initialize CLIP model."""
        model_name = config.get("model", "ViT-B/32")
        self.model, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained="openai"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def extract(self, image: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """Extract CLIP embedding from image crop."""
        if self.model is None:
            return None
            
        # Crop image
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        
        # Preprocess and get embedding
        with torch.no_grad():
            image_tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding.cpu().numpy().squeeze()
            
        return embedding
    
    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if self.model is not None:
            self.model.cpu()
            torch.cuda.empty_cache() 