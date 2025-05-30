"""
SQLite database for persistent person re-identification storage.
"""

import sqlite3
import json
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from ..reid_types import (
    TrackedPerson,
    PersonProfile,
    GlobalID,
    TrackID,
    AppearanceDescription
)
from .feature_database import BaseFeatureDatabase

class SQLiteFeatureDatabase(BaseFeatureDatabase):
    """SQLite-based feature database for persistent person profiles."""
    
    def __init__(self, config: dict):
        """
        Initialize SQLite feature database.
        
        Args:
            config: Dictionary containing:
                - database.db_path: Path to SQLite database file
                - time_decay.tau: Decay time constant in seconds
                - time_decay.min_weight: Minimum weight for old features
                - feature_matching.min_similarity: Minimum similarity threshold
                - feature_matching.required_match_percentage: Required percentage of features to match
        """
        self.config = config
        self.db_path = config["database"].get("db_path", "reid_profiles.db")
        self.conn = None
        self.initialize(config)
    
    def initialize(self, config: dict) -> None:
        """Initialize the SQLite database and create tables."""
        self.tau = config["time_decay"]["tau"]
        self.min_weight = config["time_decay"]["min_weight"]
        self.min_similarity = config["feature_matching"]["min_similarity"]
        self.required_match_percentage = config["feature_matching"]["required_match_percentage"]
        
        # Create database directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create database tables for person profiles and features."""
        cursor = self.conn.cursor()
        
        # Person profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS person_profiles (
                global_id INTEGER PRIMARY KEY,
                first_seen TIMESTAMP NOT NULL,
                last_seen TIMESTAMP NOT NULL,
                track_ids TEXT NOT NULL DEFAULT '[]'
            )
        """)
        
        # Face embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                global_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (global_id) REFERENCES person_profiles (global_id)
            )
        """)
        
        # Appearance descriptions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS appearances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                global_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                gender_guess TEXT,
                age_range TEXT,
                hair_color TEXT,
                hair_style TEXT,
                headwear_type TEXT,
                headwear_color TEXT,
                facial_features_accessories TEXT,
                upper_clothing_color_primary TEXT,
                upper_clothing_color_secondary TEXT,
                upper_clothing_type TEXT,
                upper_clothing_pattern_or_print TEXT,
                sleeve_length TEXT,
                lower_clothing_color TEXT,
                lower_clothing_type TEXT,
                lower_clothing_pattern TEXT,
                footwear_color TEXT,
                footwear_type TEXT,
                carried_items_or_prominent_accessories TEXT,
                dominant_colors_overall_outfit TEXT,
                other_distinctive_visual_cues TEXT,
                shirt_color TEXT,
                shirt_type TEXT,
                pants_color TEXT,
                pants_type TEXT,
                shoe_color TEXT,
                shoe_type TEXT,
                accessories TEXT,
                dominant_colors TEXT,
                hair TEXT,
                upper_clothing TEXT,
                lower_clothing TEXT,
                footwear TEXT,
                FOREIGN KEY (global_id) REFERENCES person_profiles (global_id)
            )
        """)
        
        # Add new columns to existing tables if they don't exist (for backward compatibility)
        new_columns = [
            ('headwear_type', 'TEXT'),
            ('headwear_color', 'TEXT'),
            ('facial_features_accessories', 'TEXT'),
            ('upper_clothing_color_primary', 'TEXT'),
            ('upper_clothing_color_secondary', 'TEXT'),
            ('upper_clothing_type', 'TEXT'),
            ('upper_clothing_pattern_or_print', 'TEXT'),
            ('sleeve_length', 'TEXT'),
            ('lower_clothing_color', 'TEXT'),
            ('lower_clothing_type', 'TEXT'),
            ('lower_clothing_pattern', 'TEXT'),
            ('footwear_color', 'TEXT'),
            ('footwear_type', 'TEXT'),
            ('carried_items_or_prominent_accessories', 'TEXT'),
            ('dominant_colors_overall_outfit', 'TEXT'),
            ('other_distinctive_visual_cues', 'TEXT')
        ]
        
        for column_name, column_type in new_columns:
            try:
                cursor.execute(f"ALTER TABLE appearances ADD COLUMN {column_name} {column_type}")
            except sqlite3.OperationalError:
                # Column already exists
                pass
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_face_global_id ON face_embeddings (global_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_appearances_global_id ON appearances (global_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_profiles_last_seen ON person_profiles (last_seen)")
        
        self.conn.commit()
    
    def add_profile(self, profile: PersonProfile) -> None:
        """Add a new person profile to the database."""
        cursor = self.conn.cursor()
        
        # Insert main profile record
        cursor.execute("""
            INSERT INTO person_profiles (global_id, first_seen, last_seen, track_ids)
            VALUES (?, ?, ?, ?)
        """, (
            profile.global_id,
            profile.first_seen,
            profile.last_seen,
            json.dumps(profile.track_ids)
        ))
        
        # Insert features
        self._insert_features(profile)
        
        self.conn.commit()
    
    def update_profile(self, profile: PersonProfile) -> None:
        """Update an existing profile with modified data."""
        cursor = self.conn.cursor()
        
        # Update main profile record
        cursor.execute("""
            UPDATE person_profiles 
            SET last_seen = ?, track_ids = ?
            WHERE global_id = ?
        """, (profile.last_seen, json.dumps(profile.track_ids), profile.global_id))
        
        # For simplicity, we'll only add new features, not update existing ones
        # In a production system, you might want to implement more sophisticated merging
        
        self.conn.commit()

    def update_profile_with_person(self, global_id: GlobalID, person: TrackedPerson) -> None:
        """Update an existing profile with new features from a tracked person."""
        cursor = self.conn.cursor()
        
        # Check if profile exists
        cursor.execute("SELECT * FROM person_profiles WHERE global_id = ?", (global_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Profile {global_id} not found")
        
        # Update last_seen and track_ids
        track_ids = json.loads(row['track_ids'])
        track_ids.append(person.track_id)
        
        cursor.execute("""
            UPDATE person_profiles 
            SET last_seen = ?, track_ids = ?
            WHERE global_id = ?
        """, (person.timestamp, json.dumps(track_ids), global_id))
        
        # Add new features
        if person.face_embedding is not None:
            cursor.execute("""
                INSERT INTO face_embeddings (global_id, timestamp, embedding)
                VALUES (?, ?, ?)
            """, (global_id, person.timestamp, pickle.dumps(person.face_embedding)))
        
        if person.appearance is not None:
            self._insert_appearance(global_id, person.timestamp, person.appearance)
        
        self.conn.commit()
    
    def find_match(self, person: TrackedPerson, min_similarity: float) -> Optional[Tuple[GlobalID, float]]:
        """Find the best matching profile for a tracked person."""
        cursor = self.conn.cursor()
        
        # Get all profiles
        cursor.execute("SELECT global_id FROM person_profiles")
        profile_ids = [row['global_id'] for row in cursor.fetchall()]
        
        if not profile_ids:
            return None
        
        best_match = None
        best_score = -1.0
        
        for global_id in profile_ids:
            profile = self.get_profile(global_id)
            if profile is None:
                continue
                
            score = self._compute_similarity(person, profile)
            if score > best_score and score >= min_similarity:
                best_score = score
                best_match = global_id
        
        return (best_match, best_score) if best_match is not None else None
    
    def cleanup_old_profiles(self, max_age: timedelta) -> None:
        """Remove profiles older than max_age."""
        cursor = self.conn.cursor()
        cutoff_time = datetime.now() - max_age
        
        # Get old profile IDs
        cursor.execute("""
            SELECT global_id FROM person_profiles 
            WHERE last_seen < ?
        """, (cutoff_time,))
        old_ids = [row['global_id'] for row in cursor.fetchall()]
        
        # Delete old profiles and their features
        for global_id in old_ids:
            cursor.execute("DELETE FROM face_embeddings WHERE global_id = ?", (global_id,))
            cursor.execute("DELETE FROM appearances WHERE global_id = ?", (global_id,))
            cursor.execute("DELETE FROM person_profiles WHERE global_id = ?", (global_id,))
        
        self.conn.commit()
    
    def get_profile(self, global_id: GlobalID) -> Optional[PersonProfile]:
        """Get a profile by global ID."""
        cursor = self.conn.cursor()
        
        # Get main profile
        cursor.execute("SELECT * FROM person_profiles WHERE global_id = ?", (global_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        # Create profile object
        profile = PersonProfile(
            global_id=row['global_id'],
            first_seen=datetime.fromisoformat(row['first_seen']),
            last_seen=datetime.fromisoformat(row['last_seen']),
            track_ids=json.loads(row['track_ids'])
        )
        
        # Load face embeddings
        cursor.execute("""
            SELECT timestamp, embedding FROM face_embeddings 
            WHERE global_id = ? ORDER BY timestamp
        """, (global_id,))
        for row in cursor.fetchall():
            timestamp = datetime.fromisoformat(row['timestamp'])
            embedding = pickle.loads(row['embedding'])
            profile.face_embeddings.append((timestamp, embedding))
        
        # Load appearances
        cursor.execute("""
            SELECT * FROM appearances 
            WHERE global_id = ? ORDER BY timestamp
        """, (global_id,))
        for row in cursor.fetchall():
            timestamp = datetime.fromisoformat(row['timestamp'])
            appearance = self._row_to_appearance(row)
            profile.appearances.append((timestamp, appearance))
        
        return profile
    
    def get_all_profiles(self) -> List[PersonProfile]:
        """Get all person profiles from the database."""
        cursor = self.conn.cursor()
        
        # Get all profile IDs
        cursor.execute("SELECT global_id FROM person_profiles ORDER BY global_id")
        profile_ids = [row['global_id'] for row in cursor.fetchall()]
        
        # Load each profile
        profiles = []
        for global_id in profile_ids:
            profile = self.get_profile(global_id)
            if profile:
                profiles.append(profile)
        
        return profiles
    
    def get_next_global_id(self) -> GlobalID:
        """Get the next available global ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(global_id) FROM person_profiles")
        row = cursor.fetchone()
        max_id = row[0] if row[0] is not None else 0
        return max_id + 1
    
    def _insert_features(self, profile: PersonProfile) -> None:
        """Insert all features for a profile."""
        cursor = self.conn.cursor()
        
        # Insert face embeddings
        for timestamp, embedding in profile.face_embeddings:
            cursor.execute("""
                INSERT INTO face_embeddings (global_id, timestamp, embedding)
                VALUES (?, ?, ?)
            """, (profile.global_id, timestamp, pickle.dumps(embedding)))
        
        # Insert appearances
        for timestamp, appearance in profile.appearances:
            self._insert_appearance(profile.global_id, timestamp, appearance)
    
    def _insert_appearance(self, global_id: GlobalID, timestamp: datetime, appearance: AppearanceDescription) -> None:
        """Insert an appearance description."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO appearances (
                global_id, timestamp, gender_guess, age_range,
                hair_color, hair_style, headwear_type, headwear_color,
                facial_features_accessories, upper_clothing_color_primary,
                upper_clothing_color_secondary, upper_clothing_type,
                upper_clothing_pattern_or_print, sleeve_length,
                lower_clothing_color, lower_clothing_type, lower_clothing_pattern,
                footwear_color, footwear_type, carried_items_or_prominent_accessories,
                dominant_colors_overall_outfit, other_distinctive_visual_cues,
                shirt_color, shirt_type, pants_color, pants_type,
                shoe_color, shoe_type, accessories, dominant_colors,
                hair, upper_clothing, lower_clothing, footwear
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            global_id, timestamp, appearance.gender_guess, appearance.age_range,
            appearance.hair_color, appearance.hair_style, appearance.headwear_type, appearance.headwear_color,
            json.dumps(appearance.facial_features_accessories) if appearance.facial_features_accessories else None,
            appearance.upper_clothing_color_primary,
            json.dumps(appearance.upper_clothing_color_secondary) if appearance.upper_clothing_color_secondary else None,
            appearance.upper_clothing_type, appearance.upper_clothing_pattern_or_print, appearance.sleeve_length,
            appearance.lower_clothing_color, appearance.lower_clothing_type, appearance.lower_clothing_pattern,
            appearance.footwear_color, appearance.footwear_type,
            json.dumps(appearance.carried_items_or_prominent_accessories) if appearance.carried_items_or_prominent_accessories else None,
            json.dumps(appearance.dominant_colors_overall_outfit) if appearance.dominant_colors_overall_outfit else None,
            appearance.other_distinctive_visual_cues,
            appearance.shirt_color, appearance.shirt_type, appearance.pants_color, appearance.pants_type,
            appearance.shoe_color, appearance.shoe_type,
            json.dumps(appearance.accessories) if appearance.accessories else None,
            json.dumps(appearance.dominant_colors) if appearance.dominant_colors else None,
            appearance.hair, appearance.upper_clothing, appearance.lower_clothing, appearance.footwear
        ))
    
    def _row_to_appearance(self, row) -> AppearanceDescription:
        """Convert database row to AppearanceDescription."""
        # Helper function to safely get column values (in case some columns don't exist in older schemas)
        def safe_get(column_name, default=None):
            try:
                return row[column_name]
            except (KeyError, IndexError):
                return default
        
        # Helper function to safely parse JSON columns
        def safe_json_get(column_name, default=None):
            if default is None:
                default = []
            try:
                value = row[column_name]
                return json.loads(value) if value else default
            except (KeyError, IndexError, json.JSONDecodeError):
                return default
        
        return AppearanceDescription(
            gender_guess=safe_get('gender_guess'),
            age_range=safe_get('age_range'),
            hair_color=safe_get('hair_color'),
            hair_style=safe_get('hair_style'),
            headwear_type=safe_get('headwear_type'),
            headwear_color=safe_get('headwear_color'),
            facial_features_accessories=safe_json_get('facial_features_accessories', []),
            upper_clothing_color_primary=safe_get('upper_clothing_color_primary'),
            upper_clothing_color_secondary=safe_json_get('upper_clothing_color_secondary', []),
            upper_clothing_type=safe_get('upper_clothing_type'),
            upper_clothing_pattern_or_print=safe_get('upper_clothing_pattern_or_print'),
            sleeve_length=safe_get('sleeve_length'),
            lower_clothing_color=safe_get('lower_clothing_color'),
            lower_clothing_type=safe_get('lower_clothing_type'),
            lower_clothing_pattern=safe_get('lower_clothing_pattern'),
            footwear_color=safe_get('footwear_color'),
            footwear_type=safe_get('footwear_type'),
            carried_items_or_prominent_accessories=safe_json_get('carried_items_or_prominent_accessories', []),
            dominant_colors_overall_outfit=safe_json_get('dominant_colors_overall_outfit', []),
            other_distinctive_visual_cues=safe_get('other_distinctive_visual_cues'),
            shirt_color=safe_get('shirt_color'),
            shirt_type=safe_get('shirt_type'),
            pants_color=safe_get('pants_color'),
            pants_type=safe_get('pants_type'),
            shoe_color=safe_get('shoe_color'),
            shoe_type=safe_get('shoe_type'),
            accessories=safe_json_get('accessories', []),
            dominant_colors=safe_json_get('dominant_colors', []),
            hair=safe_get('hair'),
            upper_clothing=safe_get('upper_clothing'),
            lower_clothing=safe_get('lower_clothing'),
            footwear=safe_get('footwear')
        )
    
    def _compute_similarity(self, person: TrackedPerson, profile: PersonProfile) -> float:
        """Compute similarity between person and profile using enhanced multi-modal matching."""
        now = datetime.now()
        similarities = []
        weights = []
        
        # Face similarity with time weighting
        if person.face_embedding is not None and profile.face_embeddings:
            face_scores = []
            face_weights = []
            
            for timestamp, embedding in profile.face_embeddings:
                time_weight = self._compute_time_weight(now - timestamp)
                similarity = self._cosine_similarity(person.face_embedding, embedding)
                
                face_scores.append(similarity)
                face_weights.append(time_weight)
            
            if face_scores:
                # Use best face match with its corresponding time weight
                best_idx = np.argmax(face_scores)
                best_face_score = face_scores[best_idx]
                best_face_weight = face_weights[best_idx]
                
                similarities.append(best_face_score)
                weights.append(best_face_weight * 0.65)  # 65% weight for face
        
        # Enhanced appearance similarity with multiple descriptions
        if person.appearance is not None and profile.appearances:
            app_scores = []
            app_weights = []
            
            for timestamp, appearance in profile.appearances:
                time_weight = self._compute_time_weight(now - timestamp)
                similarity = self._enhanced_appearance_similarity(person.appearance, appearance)
                
                app_scores.append(similarity)
                app_weights.append(time_weight)
            
            if app_scores:
                # Use weighted average of appearance scores for stability
                if len(app_scores) == 1:
                    best_app_score = app_scores[0]
                    best_app_weight = app_weights[0]
                else:
                    # Take average of top 2 scores if multiple available
                    sorted_indices = sorted(range(len(app_scores)), key=lambda i: app_scores[i], reverse=True)
                    top_scores = [app_scores[i] for i in sorted_indices[:2]]
                    top_weights = [app_weights[i] for i in sorted_indices[:2]]
                    
                    best_app_score = sum(top_scores) / len(top_scores)
                    best_app_weight = max(top_weights)
                
                similarities.append(best_app_score)
                weights.append(best_app_weight * 0.35)  # 35% weight for appearance
        
        if not similarities:
            return 0.0
        
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)
        
        # Compute weighted similarity
        final_similarity = float(np.sum(np.array(similarities) * weights))
        
        # Apply confidence adjustments based on data quality
        if len(similarities) > 1:  # Multi-modal matching bonus
            final_similarity *= 1.05
        
        if len(profile.face_embeddings) > 2 or len(profile.appearances) > 2:  # Rich profile bonus
            final_similarity *= 1.03
        
        return min(1.0, final_similarity)
    
    def _enhanced_appearance_similarity(self, a: AppearanceDescription, b: AppearanceDescription) -> float:
        """Count-based appearance similarity: count direct property matches."""
        
        # List of properties to check for exact matches
        string_properties = [
            'gender_guess', 'age_range', 'hair_color', 'hair_style',
            'headwear_type', 'headwear_color', 'upper_clothing_color_primary',
            'upper_clothing_type', 'upper_clothing_pattern_or_print', 'sleeve_length',
            'lower_clothing_color', 'lower_clothing_type', 'lower_clothing_pattern',
            'footwear_color', 'footwear_type', 'other_distinctive_visual_cues'
        ]
        
        list_properties = [
            'facial_features_accessories', 'upper_clothing_color_secondary',
            'carried_items_or_prominent_accessories', 'dominant_colors_overall_outfit'
        ]
        
        def safe_get_value(obj, prop, default="unknown"):
            """Safely get property value with default."""
            value = getattr(obj, prop, default)
            if value is None or value == "":
                return default
            return str(value).lower().strip()
        
        def safe_get_list(obj, prop, default=None):
            """Safely get list property value."""
            if default is None:
                default = []
            value = getattr(obj, prop, default)
            if value is None:
                return default
            return value if isinstance(value, list) else default
        
        def lists_have_common_items(list1, list2):
            """Check if two lists have any common items."""
            if not list1 or not list2:
                return False
            set1 = set(str(item).lower().strip() for item in list1)
            set2 = set(str(item).lower().strip() for item in list2)
            return len(set1.intersection(set2)) > 0
        
        # Count exact matches for string properties
        matches = 0
        total_checked = 0
        
        for prop in string_properties:
            val_a = safe_get_value(a, prop)
            val_b = safe_get_value(b, prop)
            
            # Only count properties that have meaningful values (not "unknown" or "none")
            if val_a not in ["unknown", "none"] and val_b not in ["unknown", "none"]:
                total_checked += 1
                if val_a == val_b:
                    matches += 1
        
        # Count matches for list properties
        for prop in list_properties:
            list_a = safe_get_list(a, prop)
            list_b = safe_get_list(b, prop)
            
            # Only count if at least one list has content
            if list_a or list_b:
                total_checked += 1
                if lists_have_common_items(list_a, list_b):
                    matches += 1
        
        # Calculate similarity as percentage of matches
        if total_checked == 0:
            return 0.0
        
        similarity = matches / total_checked
        return similarity
    
    def _compute_time_weight(self, age: timedelta) -> float:
        """Compute time decay weight for a feature."""
        weight = np.exp(-age.total_seconds() / self.tau)
        return max(weight, self.min_weight)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()