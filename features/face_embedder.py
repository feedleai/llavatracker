"""
InsightFace feature extractor for face recognition.
"""

import insightface
import numpy as np
from typing import Optional

from ..reid_types import BoundingBox
from .base import BaseFaceExtractor

class InsightFaceEmbedder(BaseFaceExtractor):
    """InsightFace feature extractor for face recognition."""
    
    def __init__(self):
        self.model = None
    
    def initialize(self, config: dict) -> None:
        """Initialize InsightFace model."""
        self.model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root="models"
        )
        self.model.prepare(ctx_id=0)  # Use GPU if available
    
    def detect_face(self, image: np.ndarray, bbox: BoundingBox) -> Optional[BoundingBox]:
        """Detect face within person bounding box."""
        if self.model is None:
            return None
            
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        faces = self.model.get(crop)
        
        if not faces:
            return None
            
        # Get largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
        
        # Convert back to original image coordinates
        return (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
    
    def extract(self, image: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """Extract face embedding."""
        if self.model is None:
            return None
            
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        faces = self.model.get(crop)
        
        if not faces:
            return None
            
        # Get embedding from largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.embedding
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.model = None 