"""
Base classes for feature extractors.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime

from ..reid_types import (
    BoundingBox,
    TrackedPerson,
    AppearanceDescription
)

class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    @abstractmethod
    def initialize(self, config: dict) -> None:
        """Initialize the feature extractor with configuration."""
        pass
    
    @abstractmethod
    def extract(self, image: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """
        Extract features from an image crop.
        
        Args:
            image: Full frame as numpy array (H, W, C)
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Feature vector as numpy array, or None if extraction failed
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (e.g., GPU memory)."""
        pass

class BaseFaceExtractor(BaseFeatureExtractor):
    """Base class for face feature extractors."""
    
    @abstractmethod
    def detect_face(self, image: np.ndarray, bbox: BoundingBox) -> Optional[BoundingBox]:
        """
        Detect face within a person bounding box.
        
        Args:
            image: Full frame as numpy array (H, W, C)
            bbox: Person bounding box (x1, y1, x2, y2)
            
        Returns:
            Face bounding box (x1, y1, x2, y2) or None if no face detected
        """
        pass

class BaseAppearanceExtractor(ABC):
    """Base class for appearance description extractors."""
    
    @abstractmethod
    def initialize(self, config: dict) -> None:
        """Initialize the appearance extractor with configuration."""
        pass
    
    @abstractmethod
    def extract(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        track_id: int
    ) -> Optional[AppearanceDescription]:
        """
        Extract appearance description from an image crop.
        
        Args:
            image: Full frame as numpy array (H, W, C)
            bbox: Bounding box (x1, y1, x2, y2)
            track_id: Track ID for logging/identification
            
        Returns:
            AppearanceDescription or None if extraction failed
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass

class FeatureExtractionPipeline:
    """Pipeline for extracting all features for a tracked person."""
    
    def __init__(
        self,
        face_extractor: Optional[BaseFaceExtractor] = None,
        appearance_extractor: Optional[BaseAppearanceExtractor] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize feature extraction pipeline.
        
        Args:
            face_extractor: Optional face feature extractor
            appearance_extractor: Optional appearance description extractor
            config: Configuration dictionary
        """
        self.face_extractor = face_extractor
        self.appearance_extractor = appearance_extractor
        self.config = config or {}
        
        # Initialize all extractors
        if self.face_extractor:
            self.face_extractor.initialize(self.config.get("face", {}))
        if self.appearance_extractor:
            self.appearance_extractor.initialize(self.config.get("llava", {}))
    
    def extract_features(
        self,
        image: np.ndarray,
        person: TrackedPerson
    ) -> TrackedPerson:
        """
        Extract all features for a tracked person.
        
        Args:
            image: Full frame as numpy array (H, W, C)
            person: TrackedPerson object to update with features
            
        Returns:
            Updated TrackedPerson with extracted features
        """
        # Extract face features if available
        if self.face_extractor:
            face_bbox = self.face_extractor.detect_face(image, person.bbox)
            if face_bbox is not None:
                face_embedding = self.face_extractor.extract(image, face_bbox)
                if face_embedding is not None:
                    person.face_embedding = face_embedding
        
        # Extract appearance description if available
        if self.appearance_extractor:
            appearance = self.appearance_extractor.extract(
                image, person.bbox, person.track_id
            )
            if appearance is not None:
                person.appearance = appearance
        
        return person
    
    def cleanup(self) -> None:
        """Clean up all extractors."""
        if self.face_extractor:
            self.face_extractor.cleanup()
        if self.appearance_extractor:
            self.appearance_extractor.cleanup() 