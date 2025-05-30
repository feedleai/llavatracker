"""
Type definitions and data structures for the hybrid re-identification system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

# Basic types
BoundingBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
TrackID = int
GlobalID = int
FrameID = int
Confidence = float

class AppearanceDescription(BaseModel):
    """Structured appearance description from LLaVA with detailed color and style information."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Basic information
    gender_guess: Optional[str] = None
    age_range: Optional[str] = None
    
    # Hair details
    hair_color: Optional[str] = None
    hair_style: Optional[str] = None
    
    # Headwear details
    headwear_type: Optional[str] = None
    headwear_color: Optional[str] = None
    
    # Facial accessories
    facial_features_accessories: Optional[List[str]] = None
    
    # Upper clothing details
    upper_clothing_color_primary: Optional[str] = None
    upper_clothing_color_secondary: Optional[List[str]] = None
    upper_clothing_type: Optional[str] = None
    upper_clothing_pattern_or_print: Optional[str] = None
    sleeve_length: Optional[str] = None
    
    # Lower clothing details
    lower_clothing_color: Optional[str] = None
    lower_clothing_type: Optional[str] = None
    lower_clothing_pattern: Optional[str] = None
    
    # Footwear details
    footwear_color: Optional[str] = None
    footwear_type: Optional[str] = None
    
    # Accessories and distinctive features
    carried_items_or_prominent_accessories: Optional[List[str]] = None
    dominant_colors_overall_outfit: Optional[List[str]] = None
    other_distinctive_visual_cues: Optional[str] = None
    
    # Legacy fields for backward compatibility (kept for older database entries)
    shirt_color: Optional[str] = None
    shirt_type: Optional[str] = None
    pants_color: Optional[str] = None
    pants_type: Optional[str] = None
    shoe_color: Optional[str] = None
    shoe_type: Optional[str] = None
    accessories: Optional[List[str]] = None
    dominant_colors: Optional[List[str]] = None
    hair: Optional[str] = None
    upper_clothing: Optional[str] = None
    lower_clothing: Optional[str] = None
    footwear: Optional[str] = None

@dataclass
class TrackedPerson:
    """Represents a tracked person in a single frame."""
    track_id: TrackID
    bbox: BoundingBox
    confidence: Confidence
    frame_id: FrameID
    timestamp: datetime
    global_id: Optional[GlobalID] = None
    face_embedding: Optional[np.ndarray] = None
    appearance: Optional[AppearanceDescription] = None

class PersonProfile(BaseModel):
    """Persistent profile for a person across frames."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    global_id: GlobalID
    first_seen: datetime
    last_seen: datetime
    track_ids: List[TrackID] = Field(default_factory=list)
    face_embeddings: List[Tuple[datetime, np.ndarray]] = Field(default_factory=list)
    appearances: List[Tuple[datetime, AppearanceDescription]] = Field(default_factory=list)
    
    def add_features(
        self,
        timestamp: datetime,
        face_embedding: Optional[np.ndarray] = None,
        appearance: Optional[AppearanceDescription] = None
    ) -> None:
        """Add new features to the profile with timestamp."""
        if face_embedding is not None:
            self.face_embeddings.append((timestamp, face_embedding))
        if appearance is not None:
            self.appearances.append((timestamp, appearance))
        self.last_seen = timestamp

class TrackingResult(BaseModel):
    """Results from tracking a single frame."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    frame_id: FrameID
    timestamp: datetime
    tracked_persons: List[TrackedPerson]
    frame: np.ndarray  # Original frame

class ReIDResult(BaseModel):
    """Results from re-identification processing."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    frame_id: FrameID
    timestamp: datetime
    assignments: Dict[TrackID, GlobalID]  # Track ID -> Global ID mapping
    new_global_ids: List[GlobalID]  # Newly created global IDs
    reused_global_ids: List[GlobalID]  # Reused global IDs
    frame: np.ndarray  # Annotated frame with IDs 