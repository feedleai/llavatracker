"""
ID resolver for mapping track IDs to global IDs.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..types import (
    TrackedPerson,
    PersonProfile,
    GlobalID,
    TrackID,
    ReIDResult
)

logger = logging.getLogger(__name__)

class IDResolver:
    """
    Resolves track IDs to global IDs using feature matching.
    Maintains a mapping between track IDs and global IDs.
    """
    
    def __init__(self, config: dict):
        """
        Initialize ID resolver.
        
        Args:
            config: Dictionary containing:
                - feature_matching.min_similarity: Minimum similarity threshold
                - feature_matching.required_match_percentage: Required percentage of features to match
                - time_decay.tau: Decay time constant in seconds
        """
        self.config = config
        self.track_to_global: Dict[TrackID, GlobalID] = {}
        self.next_global_id = 0
        self.min_similarity = config["feature_matching"]["min_similarity"]
        self.required_match_percentage = config["feature_matching"]["required_match_percentage"]
    
    def resolve_ids(
        self,
        tracked_persons: List[TrackedPerson],
        feature_db: 'TimeDecayFeatureDatabase',  # Forward reference
        frame_id: int,
        timestamp: datetime
    ) -> ReIDResult:
        """
        Resolve track IDs to global IDs for a frame.
        
        Args:
            tracked_persons: List of tracked persons in the frame
            feature_db: Feature database for matching
            frame_id: Current frame ID
            timestamp: Current frame timestamp
            
        Returns:
            ReIDResult containing ID assignments and frame
        """
        assignments: Dict[TrackID, GlobalID] = {}
        new_global_ids: List[GlobalID] = []
        reused_global_ids: List[GlobalID] = []
        
        for person in tracked_persons:
            # Check if we already have a global ID for this track
            if person.track_id in self.track_to_global:
                global_id = self.track_to_global[person.track_id]
                assignments[person.track_id] = global_id
                reused_global_ids.append(global_id)
                
                # Update profile with new features
                feature_db.update_profile(global_id, person)
                continue
            
            # Try to find a match in the database
            match = feature_db.find_match(person, self.min_similarity)
            
            if match is not None:
                global_id, similarity = match
                logger.info(
                    f"Matched track {person.track_id} to global {global_id} "
                    f"with similarity {similarity:.3f}"
                )
                assignments[person.track_id] = global_id
                reused_global_ids.append(global_id)
                
                # Update profile with new features
                feature_db.update_profile(global_id, person)
            else:
                # Create new global ID
                global_id = self._get_next_global_id()
                assignments[person.track_id] = global_id
                new_global_ids.append(global_id)
                
                # Create new profile
                profile = PersonProfile(
                    global_id=global_id,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    track_ids=[person.track_id]
                )
                profile.add_features(
                    timestamp=timestamp,
                    clip_embedding=person.clip_embedding,
                    face_embedding=person.face_embedding,
                    appearance=person.appearance
                )
                feature_db.add_profile(profile)
                
                logger.info(f"Created new global ID {global_id} for track {person.track_id}")
            
            # Update track-to-global mapping
            self.track_to_global[person.track_id] = global_id
        
        # Clean up old track IDs
        self._cleanup_old_tracks(timestamp)
        
        return ReIDResult(
            frame_id=frame_id,
            timestamp=timestamp,
            assignments=assignments,
            new_global_ids=new_global_ids,
            reused_global_ids=reused_global_ids,
            frame=tracked_persons[0].frame if tracked_persons else None
        )
    
    def _get_next_global_id(self) -> GlobalID:
        """Get the next available global ID."""
        global_id = self.next_global_id
        self.next_global_id += 1
        return global_id
    
    def _cleanup_old_tracks(self, current_time: datetime) -> None:
        """
        Remove track IDs that haven't been seen for a while.
        This helps prevent memory leaks from abandoned tracks.
        """
        # TODO: Implement track cleanup logic
        # For now, we keep all track IDs
        pass
    
    def get_global_id(self, track_id: TrackID) -> Optional[GlobalID]:
        """Get the global ID for a track ID if it exists."""
        return self.track_to_global.get(track_id)
    
    def get_track_ids(self, global_id: GlobalID) -> List[TrackID]:
        """Get all track IDs associated with a global ID."""
        return [
            track_id for track_id, gid in self.track_to_global.items()
            if gid == global_id
        ] 