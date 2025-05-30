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
    Resolves track IDs to global IDs using feature matching with 80% similarity threshold.
    Maintains a mapping between track IDs and global IDs and handles re-identification
    when tracking is lost.
    """
    
    def __init__(self, config: dict):
        """
        Initialize ID resolver.
        
        Args:
            config: Dictionary containing:
                - feature_matching.min_similarity: Minimum similarity threshold (0.8 for 80%)
                - feature_matching.required_match_percentage: Required percentage of features to match
                - time_decay.tau: Decay time constant in seconds
        """
        self.config = config
        self.track_to_global: Dict[TrackID, GlobalID] = {}
        self.next_global_id = 1  # Start from 1 for nicer display (P1, P2, etc.)
        # Use 0.8 (80%) as minimum similarity for robust re-identification
        self.min_similarity = config["feature_matching"].get("min_similarity", 0.8)
        self.required_match_percentage = config["feature_matching"].get("required_match_percentage", 0.8)
        
        # Track lost persons for potential re-identification
        self.lost_tracks: Dict[TrackID, datetime] = {}
        self.track_timeout = timedelta(seconds=config.get("track_timeout", 30))
    
    def resolve_ids(
        self,
        tracked_persons: List[TrackedPerson],
        feature_db,  # SQLiteFeatureDatabase
        frame_id: int,
        timestamp: datetime
    ) -> ReIDResult:
        """
        Resolve track IDs to global IDs for a frame with enhanced re-identification.
        
        Args:
            tracked_persons: List of tracked persons in the frame
            feature_db: SQLite feature database for matching
            frame_id: Current frame ID
            timestamp: Current frame timestamp
            
        Returns:
            ReIDResult containing ID assignments and frame
        """
        assignments: Dict[TrackID, GlobalID] = {}
        new_global_ids: List[GlobalID] = []
        reused_global_ids: List[GlobalID] = []
        
        # Get current track IDs to identify lost tracks
        current_track_ids = {person.track_id for person in tracked_persons}
        self._update_lost_tracks(current_track_ids, timestamp)
        
        for person in tracked_persons:
            # Check if we already have a global ID for this track
            if person.track_id in self.track_to_global:
                global_id = self.track_to_global[person.track_id]
                assignments[person.track_id] = global_id
                reused_global_ids.append(global_id)
                
                # Update profile with new features
                feature_db.update_profile(global_id, person)
                continue
            
            # Try to find a match in the database (compare against ALL stored profiles)
            match = feature_db.find_match(person, self.min_similarity)
            
            if match is not None:
                global_id, similarity = match
                logger.info(
                    f"Re-identified person: track {person.track_id} matched to global {global_id} "
                    f"with {similarity:.1%} similarity"
                )
                assignments[person.track_id] = global_id
                reused_global_ids.append(global_id)
                
                # Update profile with new features
                feature_db.update_profile(global_id, person)
                
                # Remove from lost tracks if it was there
                if person.track_id in self.lost_tracks:
                    del self.lost_tracks[person.track_id]
            else:
                # Create new global ID - this is a genuinely new person
                global_id = self._get_next_global_id(feature_db)
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
                    face_embedding=person.face_embedding,
                    appearance=person.appearance
                )
                feature_db.add_profile(profile)
                
                logger.info(f"New person detected: assigned global ID P{global_id} to track {person.track_id}")
            
            # Update track-to-global mapping
            self.track_to_global[person.track_id] = global_id
        
        # Clean up old track mappings
        self._cleanup_old_tracks(timestamp)
        
        return ReIDResult(
            frame_id=frame_id,
            timestamp=timestamp,
            assignments=assignments,
            new_global_ids=new_global_ids,
            reused_global_ids=reused_global_ids,
            frame=tracked_persons[0].frame if tracked_persons else None
        )
    
    def _get_next_global_id(self, feature_db) -> GlobalID:
        """Get the next available global ID from database."""
        # Use database method to get next ID to ensure consistency
        if hasattr(feature_db, 'get_next_global_id'):
            return feature_db.get_next_global_id()
        else:
            # Fallback for compatibility
            global_id = self.next_global_id
            self.next_global_id += 1
            return global_id
    
    def _update_lost_tracks(self, current_track_ids: set, timestamp: datetime) -> None:
        """Update the list of lost tracks."""
        # Find tracks that disappeared
        all_known_tracks = set(self.track_to_global.keys())
        lost_track_ids = all_known_tracks - current_track_ids
        
        # Add newly lost tracks
        for track_id in lost_track_ids:
            if track_id not in self.lost_tracks:
                self.lost_tracks[track_id] = timestamp
                logger.debug(f"Track {track_id} lost at {timestamp}")
    
    def _cleanup_old_tracks(self, current_time: datetime) -> None:
        """
        Remove track IDs that haven't been seen for a while to prevent memory leaks.
        """
        # Remove tracks that have been lost for too long
        tracks_to_remove = [
            track_id for track_id, lost_time in self.lost_tracks.items()
            if (current_time - lost_time) > self.track_timeout
        ]
        
        for track_id in tracks_to_remove:
            if track_id in self.track_to_global:
                global_id = self.track_to_global[track_id]
                logger.debug(f"Cleaning up old track {track_id} (global ID P{global_id})")
                del self.track_to_global[track_id]
            del self.lost_tracks[track_id]
    
    def get_global_id(self, track_id: TrackID) -> Optional[GlobalID]:
        """Get the global ID for a track ID if it exists."""
        return self.track_to_global.get(track_id)
    
    def get_track_ids(self, global_id: GlobalID) -> List[TrackID]:
        """Get all track IDs associated with a global ID."""
        return [
            track_id for track_id, gid in self.track_to_global.items()
            if gid == global_id
        ]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the ID resolver state."""
        return {
            "active_tracks": len(self.track_to_global),
            "lost_tracks": len(self.lost_tracks),
            "next_global_id": self.next_global_id,
            "unique_persons": len(set(self.track_to_global.values()))
        } 