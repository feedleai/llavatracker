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

from ..types import (
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
    
    def update_profile(self, global_id: GlobalID, person: TrackedPerson) -> None:
        """Update an existing profile with new features."""
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
                hair_color, hair_style, shirt_color, shirt_type,
                pants_color, pants_type, shoe_color, shoe_type,
                accessories, dominant_colors, hair, upper_clothing,
                lower_clothing, footwear
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            global_id, timestamp, appearance.gender_guess, appearance.age_range,
            appearance.hair_color, appearance.hair_style, appearance.shirt_color, appearance.shirt_type,
            appearance.pants_color, appearance.pants_type, appearance.shoe_color, appearance.shoe_type,
            json.dumps(appearance.accessories) if appearance.accessories else None,
            json.dumps(appearance.dominant_colors) if appearance.dominant_colors else None,
            appearance.hair, appearance.upper_clothing, appearance.lower_clothing, appearance.footwear
        ))
    
    def _row_to_appearance(self, row) -> AppearanceDescription:
        """Convert database row to AppearanceDescription."""
        return AppearanceDescription(
            gender_guess=row['gender_guess'],
            age_range=row['age_range'],
            hair_color=row['hair_color'],
            hair_style=row['hair_style'],
            shirt_color=row['shirt_color'],
            shirt_type=row['shirt_type'],
            pants_color=row['pants_color'],
            pants_type=row['pants_type'],
            shoe_color=row['shoe_color'],
            shoe_type=row['shoe_type'],
            accessories=json.loads(row['accessories']) if row['accessories'] else None,
            dominant_colors=json.loads(row['dominant_colors']) if row['dominant_colors'] else None,
            hair=row['hair'],
            upper_clothing=row['upper_clothing'],
            lower_clothing=row['lower_clothing'],
            footwear=row['footwear']
        )
    
    def _compute_similarity(self, person: TrackedPerson, profile: PersonProfile) -> float:
        """Compute similarity between person and profile using enhanced appearance matching."""
        now = datetime.now()
        similarities = []
        weights = []
        
        # Compare face embeddings
        if person.face_embedding is not None and profile.face_embeddings:
            for timestamp, embedding in profile.face_embeddings:
                weight = self._compute_time_weight(now - timestamp)
                similarity = self._cosine_similarity(person.face_embedding, embedding)
                similarities.append(similarity)
                weights.append(weight * 0.6)  # 60% weight for face
        
        # Compare appearance descriptions with enhanced matching
        if person.appearance is not None and profile.appearances:
            for timestamp, appearance in profile.appearances:
                weight = self._compute_time_weight(now - timestamp)
                similarity = self._enhanced_appearance_similarity(person.appearance, appearance)
                similarities.append(similarity)
                weights.append(weight * 0.4)  # 40% weight for appearance
        
        if not similarities:
            return 0.0
        
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)
        
        # Compute weighted average similarity
        return float(np.sum(np.array(similarities) * weights))
    
    def _enhanced_appearance_similarity(self, a: AppearanceDescription, b: AppearanceDescription) -> float:
        """Enhanced appearance similarity with specific color and style matching."""
        matches = 0
        total = 0
        
        # Weight different features differently
        feature_weights = {
            'hair_color': 0.15,
            'shirt_color': 0.20,
            'pants_color': 0.20,
            'shoe_color': 0.15,
            'shirt_type': 0.10,
            'pants_type': 0.10,
            'gender_guess': 0.10
        }
        
        total_weight = 0
        weighted_score = 0
        
        for field, weight in feature_weights.items():
            val_a = getattr(a, field, None)
            val_b = getattr(b, field, None)
            
            if val_a is not None and val_b is not None and val_a != "unknown" and val_b != "unknown":
                total_weight += weight
                if self._color_match(val_a, val_b):
                    weighted_score += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _color_match(self, color1: str, color2: str) -> bool:
        """Enhanced color matching with fuzzy matching for similar colors."""
        if color1.lower() == color2.lower():
            return True
        
        # Define color groups for fuzzy matching
        color_groups = [
            ['black', 'dark', 'charcoal'],
            ['white', 'light', 'cream', 'ivory'],
            ['blue', 'navy', 'dark blue', 'light blue'],
            ['red', 'crimson', 'dark red', 'light red'],
            ['green', 'dark green', 'light green'],
            ['brown', 'tan', 'beige', 'khaki'],
            ['gray', 'grey', 'silver'],
            ['yellow', 'gold', 'blonde']
        ]
        
        color1_lower = color1.lower()
        color2_lower = color2.lower()
        
        for group in color_groups:
            if any(c in color1_lower for c in group) and any(c in color2_lower for c in group):
                return True
        
        return False
    
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