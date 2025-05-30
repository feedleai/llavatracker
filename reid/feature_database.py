"""
Feature database for person re-identification.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from ..reid_types import (
    TrackedPerson,
    PersonProfile,
    GlobalID,
    TrackID,
    AppearanceDescription
)

class BaseFeatureDatabase(ABC):
    """Abstract base class for feature databases."""
    
    @abstractmethod
    def initialize(self, config: dict) -> None:
        """Initialize the database with configuration."""
        pass
    
    @abstractmethod
    def add_profile(self, profile: PersonProfile) -> None:
        """Add a new person profile to the database."""
        pass
    
    @abstractmethod
    def update_profile(
        self,
        global_id: GlobalID,
        person: TrackedPerson
    ) -> None:
        """Update an existing profile with new features."""
        pass
    
    @abstractmethod
    def find_match(
        self,
        person: TrackedPerson,
        min_similarity: float
    ) -> Optional[Tuple[GlobalID, float]]:
        """
        Find the best matching profile for a tracked person.
        
        Args:
            person: TrackedPerson to match
            min_similarity: Minimum similarity threshold
            
        Returns:
            Tuple of (GlobalID, similarity_score) or None if no match
        """
        pass
    
    @abstractmethod
    def cleanup_old_profiles(self, max_age: timedelta) -> None:
        """Remove profiles older than max_age."""
        pass
    
    @abstractmethod
    def get_profile(self, global_id: GlobalID) -> Optional[PersonProfile]:
        """Get a profile by global ID."""
        pass

class TimeDecayFeatureDatabase(BaseFeatureDatabase):
    """Feature database with time-decay weighted matching."""
    
    def __init__(self, config: dict):
        """
        Initialize time-decay feature database.
        
        Args:
            config: Dictionary containing:
                - time_decay.tau: Decay time constant in seconds
                - time_decay.min_weight: Minimum weight for old features
                - feature_matching.min_similarity: Minimum similarity threshold
                - feature_matching.required_match_percentage: Required percentage of features to match
                - database.max_profiles: Maximum number of profiles to store
        """
        self.config = config
        self.profiles: Dict[GlobalID, PersonProfile] = {}
        self.next_global_id = 0
        self.initialize(config)
    
    def initialize(self, config: dict) -> None:
        """Initialize the database with configuration."""
        self.tau = config["time_decay"]["tau"]
        self.min_weight = config["time_decay"]["min_weight"]
        self.min_similarity = config["feature_matching"]["min_similarity"]
        self.required_match_percentage = config["feature_matching"]["required_match_percentage"]
        self.max_profiles = config["database"]["max_profiles"]
    
    def add_profile(self, profile: PersonProfile) -> None:
        """Add a new person profile to the database."""
        if len(self.profiles) >= self.max_profiles:
            # Remove oldest profile if at capacity
            oldest_id = min(
                self.profiles.keys(),
                key=lambda x: self.profiles[x].last_seen
            )
            del self.profiles[oldest_id]
        
        self.profiles[profile.global_id] = profile
    
    def update_profile(
        self,
        global_id: GlobalID,
        person: TrackedPerson
    ) -> None:
        """Update an existing profile with new features."""
        if global_id not in self.profiles:
            raise ValueError(f"Profile {global_id} not found")
        
        profile = self.profiles[global_id]
        profile.add_features(
            timestamp=person.timestamp,
            clip_embedding=person.clip_embedding,
            face_embedding=person.face_embedding,
            appearance=person.appearance
        )
        profile.track_ids.append(person.track_id)
    
    def find_match(
        self,
        person: TrackedPerson,
        min_similarity: float
    ) -> Optional[Tuple[GlobalID, float]]:
        """
        Find the best matching profile using time-decay weighted matching.
        
        Args:
            person: TrackedPerson to match
            min_similarity: Minimum similarity threshold
            
        Returns:
            Tuple of (GlobalID, similarity_score) or None if no match
        """
        if not self.profiles:
            return None
        
        best_match = None
        best_score = -1.0
        
        for global_id, profile in self.profiles.items():
            score = self._compute_similarity(person, profile)
            if score > best_score and score >= min_similarity:
                best_score = score
                best_match = global_id
        
        return (best_match, best_score) if best_match is not None else None
    
    def cleanup_old_profiles(self, max_age: timedelta) -> None:
        """Remove profiles older than max_age."""
        now = datetime.now()
        to_remove = [
            global_id for global_id, profile in self.profiles.items()
            if (now - profile.last_seen) > max_age
        ]
        for global_id in to_remove:
            del self.profiles[global_id]
    
    def get_profile(self, global_id: GlobalID) -> Optional[PersonProfile]:
        """Get a profile by global ID."""
        return self.profiles.get(global_id)
    
    def _compute_similarity(
        self,
        person: TrackedPerson,
        profile: PersonProfile
    ) -> float:
        """
        Compute time-decay weighted similarity between person and profile.
        
        Args:
            person: TrackedPerson to match
            profile: PersonProfile to match against
            
        Returns:
            Similarity score between 0 and 1
        """
        now = datetime.now()
        similarities = []
        weights = []
        
        # Compare CLIP embeddings
        if person.clip_embedding is not None and profile.clip_embeddings:
            for timestamp, embedding in profile.clip_embeddings:
                weight = self._compute_time_weight(now - timestamp)
                similarity = self._cosine_similarity(
                    person.clip_embedding,
                    embedding
                )
                similarities.append(similarity)
                weights.append(weight)
        
        # Compare face embeddings
        if person.face_embedding is not None and profile.face_embeddings:
            for timestamp, embedding in profile.face_embeddings:
                weight = self._compute_time_weight(now - timestamp)
                similarity = self._cosine_similarity(
                    person.face_embedding,
                    embedding
                )
                similarities.append(similarity)
                weights.append(weight)
        
        # Compare appearance descriptions
        if person.appearance is not None and profile.appearances:
            for timestamp, appearance in profile.appearances:
                weight = self._compute_time_weight(now - timestamp)
                similarity = self._appearance_similarity(
                    person.appearance,
                    appearance
                )
                similarities.append(similarity)
                weights.append(weight)
        
        if not similarities:
            return 0.0
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Compute weighted average similarity
        return float(np.sum(np.array(similarities) * weights))
    
    def _compute_time_weight(self, age: timedelta) -> float:
        """Compute time decay weight for a feature."""
        weight = np.exp(-age.total_seconds() / self.tau)
        return max(weight, self.min_weight)
    
    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _appearance_similarity(
        self,
        a: AppearanceDescription,
        b: AppearanceDescription
    ) -> float:
        """
        Compute similarity between two appearance descriptions.
        This is a simple implementation that can be improved.
        """
        matches = 0
        total = 0
        
        for field in a.__fields__:
            if field == "accessories":
                continue  # Skip accessories for now
            
            val_a = getattr(a, field)
            val_b = getattr(b, field)
            
            if val_a is not None and val_b is not None:
                total += 1
                if val_a == val_b:
                    matches += 1
        
        return matches / total if total > 0 else 0.0 