"""
Type definitions and data structures for the hybrid re-identification system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

# Basic types
BoundingBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
TrackID = int
GlobalID = int
FrameID = int
Confidence = float

class AppearanceDescription(BaseModel):
    """Structured appearance description from LLaVA."""
    gender_guess: Optional[str] = None
    age_range: Optional[str] = None
    hair: Optional[str] = None
    upper_clothing: Optional[str] = None
    lower_clothing: Optional[str] = None
    footwear: Optional[str] = None
    accessories: Optional[List[str]] = None

@dataclass
class TrackedPerson:
    """Represents a tracked person in a single frame."""
    track_id: TrackID
    bbox: BoundingBox
    confidence: Confidence
    frame_id: FrameID
    timestamp: datetime
    global_id: Optional[GlobalID] = None
    clip_embedding: Optional[np.ndarray] = None
    face_embedding: Optional[np.ndarray] = None
    appearance: Optional[AppearanceDescription] = None

class PersonProfile(BaseModel):
    """Persistent profile for a person across frames."""
    global_id: GlobalID
    first_seen: datetime
    last_seen: datetime
    track_ids: List[TrackID] = Field(default_factory=list)
    clip_embeddings: List[Tuple[datetime, np.ndarray]] = Field(default_factory=list)
    face_embeddings: List[Tuple[datetime, np.ndarray]] = Field(default_factory=list)
    appearances: List[Tuple[datetime, AppearanceDescription]] = Field(default_factory=list)
    
    def add_features(
        self,
        timestamp: datetime,
        clip_embedding: Optional[np.ndarray] = None,
        face_embedding: Optional[np.ndarray] = None,
        appearance: Optional[AppearanceDescription] = None
    ) -> None:
        """Add new features to the profile with timestamp."""
        if clip_embedding is not None:
            self.clip_embeddings.append((timestamp, clip_embedding))
        if face_embedding is not None:
            self.face_embeddings.append((timestamp, face_embedding))
        if appearance is not None:
            self.appearances.append((timestamp, appearance))
        self.last_seen = timestamp

class TrackingResult(BaseModel):
    """Results from tracking a single frame."""
    frame_id: FrameID
    timestamp: datetime
    tracked_persons: List[TrackedPerson]
    frame: np.ndarray  # Original frame

class ReIDResult(BaseModel):
    """Results from re-identification processing."""
    frame_id: FrameID
    timestamp: datetime
    assignments: Dict[TrackID, GlobalID]  # Track ID -> Global ID mapping
    new_global_ids: List[GlobalID]  # Newly created global IDs
    reused_global_ids: List[GlobalID]  # Reused global IDs
    frame: np.ndarray  # Annotated frame with IDs 