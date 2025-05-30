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
        """Compute similarity between person and profile using enhanced multi-modal matching."""
        now = datetime.now()
        similarities = []
        weights = []
        
        # Enhanced face similarity with multiple embeddings
        if person.face_embedding is not None and profile.face_embeddings:
            face_scores = []
            face_weights = []
            
            for timestamp, embedding in profile.face_embeddings:
                time_weight = self._compute_time_weight(now - timestamp)
                similarity = self._cosine_similarity(person.face_embedding, embedding)
                
                # Apply confidence boost for high-quality face matches
                if similarity > 0.85:
                    similarity = min(1.0, similarity * 1.1)  # Boost very high similarities
                
                face_scores.append(similarity)
                face_weights.append(time_weight)
            
            if face_scores:
                # Use the best face match (not average) for robustness
                best_face_score = max(face_scores)
                best_face_weight = max(face_weights)
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
        """Enhanced appearance similarity with robust color and semantic matching."""
        
        # Comprehensive color mapping for robust matching
        color_families = {
            'black': ['black', 'dark', 'charcoal', 'ebony', 'jet', 'coal'],
            'white': ['white', 'light', 'cream', 'ivory', 'off-white', 'pearl', 'snow'],
            'gray': ['gray', 'grey', 'silver', 'ash', 'slate', 'charcoal', 'gunmetal'],
            'blue': ['blue', 'navy', 'royal', 'cobalt', 'azure', 'cerulean', 'sapphire', 'denim'],
            'red': ['red', 'crimson', 'scarlet', 'burgundy', 'maroon', 'cherry', 'wine', 'rust'],
            'green': ['green', 'emerald', 'forest', 'olive', 'lime', 'mint', 'sage', 'jade'],
            'brown': ['brown', 'tan', 'beige', 'khaki', 'camel', 'chocolate', 'coffee', 'mocha'],
            'yellow': ['yellow', 'gold', 'golden', 'amber', 'blonde', 'cream', 'butter'],
            'orange': ['orange', 'coral', 'peach', 'rust', 'bronze', 'copper'],
            'purple': ['purple', 'violet', 'lavender', 'plum', 'magenta', 'mauve'],
            'pink': ['pink', 'rose', 'salmon', 'blush', 'fuchsia']
        }
        
        def robust_color_match(color1: str, color2: str) -> float:
            """Robust color matching with fuzzy logic."""
            if not color1 or not color2 or color1 == "unknown" or color2 == "unknown":
                return 0.0
            
            c1_clean = color1.lower().strip()
            c2_clean = color2.lower().strip()
            
            # Exact match
            if c1_clean == c2_clean:
                return 1.0
            
            # Color family matching
            for family, variants in color_families.items():
                c1_in_family = any(variant in c1_clean for variant in variants)
                c2_in_family = any(variant in c2_clean for variant in variants)
                
                if c1_in_family and c2_in_family:
                    # Same family - check for exact variant match
                    for variant in variants:
                        if variant in c1_clean and variant in c2_clean:
                            return 0.95  # Very close match within family
                    return 0.8  # Same family, different variants
            
            # Partial word matching
            words1 = set(c1_clean.split())
            words2 = set(c2_clean.split())
            intersection = words1.intersection(words2)
            
            if intersection:
                overlap_ratio = len(intersection) / max(len(words1), len(words2))
                return overlap_ratio * 0.6
            
            return 0.0
        
        def clothing_semantic_match(type1: str, type2: str) -> float:
            """Semantic matching for clothing types."""
            if not type1 or not type2 or type1 == "unknown" or type2 == "unknown":
                return 0.0
            
            t1_clean = type1.lower().strip()
            t2_clean = type2.lower().strip()
            
            if t1_clean == t2_clean:
                return 1.0
            
            # Clothing semantic groups
            semantic_groups = {
                'casual_shirts': ['t-shirt', 'tee', 'polo', 'tank-top', 'casual'],
                'formal_shirts': ['shirt', 'dress-shirt', 'button-down', 'formal', 'blouse'],
                'outerwear': ['jacket', 'hoodie', 'sweater', 'cardigan', 'coat', 'blazer'],
                'pants_casual': ['jeans', 'denim', 'casual', 'chinos'],
                'pants_formal': ['trousers', 'dress-pants', 'slacks', 'formal'],
                'pants_athletic': ['shorts', 'athletic', 'sport', 'gym', 'sweatpants'],
                'dresses': ['dress', 'gown', 'frock', 'sundress'],
                'skirts': ['skirt', 'mini', 'maxi', 'midi'],
                'shoes_athletic': ['sneakers', 'trainers', 'athletic', 'sport', 'running'],
                'shoes_formal': ['dress-shoes', 'oxford', 'formal', 'loafers', 'heels'],
                'shoes_casual': ['boots', 'sandals', 'flats', 'casual']
            }
            
            # Check semantic similarity
            for group, items in semantic_groups.items():
                t1_in_group = any(item in t1_clean for item in items)
                t2_in_group = any(item in t2_clean for item in items)
                
                if t1_in_group and t2_in_group:
                    # Check for exact item match within group
                    for item in items:
                        if item in t1_clean and item in t2_clean:
                            return 0.9  # Very similar within category
                    return 0.7  # Same category, different types
            
            return 0.0
        
        # Adaptive feature weights based on distinctiveness and reliability
        feature_weights = {
            'gender_guess': {'weight': 1.5, 'critical': True, 'type': 'exact'},
            'age_range': {'weight': 1.0, 'critical': False, 'type': 'exact'},
            'hair_color': {'weight': 2.2, 'critical': True, 'type': 'color'},
            'hair_style': {'weight': 1.3, 'critical': False, 'type': 'semantic'},
            'shirt_color': {'weight': 3.0, 'critical': True, 'type': 'color'},
            'shirt_type': {'weight': 2.0, 'critical': True, 'type': 'semantic'},
            'pants_color': {'weight': 2.8, 'critical': True, 'type': 'color'},
            'pants_type': {'weight': 1.8, 'critical': True, 'type': 'semantic'},
            'shoe_color': {'weight': 1.6, 'critical': False, 'type': 'color'},
            'shoe_type': {'weight': 1.4, 'critical': False, 'type': 'semantic'}
        }
        
        total_weight = 0
        weighted_score = 0
        available_features = 0
        critical_matches = 0
        critical_features = 0
        
        # Compare individual features
        for field, config in feature_weights.items():
            val_a = getattr(a, field, None)
            val_b = getattr(b, field, None)
            
            if val_a and val_b and val_a != "unknown" and val_b != "unknown":
                available_features += 1
                weight = config['weight']
                
                if config['critical']:
                    critical_features += 1
                
                # Calculate similarity based on feature type
                if config['type'] == 'color':
                    similarity = robust_color_match(val_a, val_b)
                elif config['type'] == 'semantic':
                    similarity = clothing_semantic_match(val_a, val_b)
                else:  # exact match
                    similarity = 1.0 if val_a.lower() == val_b.lower() else 0.0
                
                # Track critical feature matches
                if similarity > 0.6 and config['critical']:
                    critical_matches += 1
                
                weighted_score += similarity * weight
                total_weight += weight
        
        # Enhanced accessories matching
        acc_a = set(getattr(a, 'accessories', []) or [])
        acc_b = set(getattr(b, 'accessories', []) or [])
        
        if acc_a or acc_b:
            if acc_a and acc_b:
                jaccard = len(acc_a.intersection(acc_b)) / len(acc_a.union(acc_b))
                weighted_score += jaccard * 1.5
            total_weight += 1.5
        
        # Enhanced dominant colors matching
        colors_a = getattr(a, 'dominant_colors', []) or []
        colors_b = getattr(b, 'dominant_colors', []) or []
        
        if colors_a and colors_b:
            color_similarity_sum = 0
            total_comparisons = 0
            
            for ca in colors_a:
                best_match = 0
                for cb in colors_b:
                    match_score = robust_color_match(ca, cb)
                    best_match = max(best_match, match_score)
                color_similarity_sum += best_match
                total_comparisons += 1
            
            if total_comparisons > 0:
                avg_color_sim = color_similarity_sum / total_comparisons
                weighted_score += avg_color_sim * 2.2
            total_weight += 2.2
        
        if total_weight == 0:
            return 0.0
        
        base_similarity = weighted_score / total_weight
        
        # Apply quality-based adjustments
        
        # Critical features bonus - if most distinctive features match well
        if critical_features > 0:
            critical_ratio = critical_matches / critical_features
            if critical_ratio >= 0.6:  # At least 60% of critical features match
                base_similarity *= (1.0 + 0.15 * critical_ratio)  # Up to 15% bonus
        
        # Data quality penalty for insufficient information
        if available_features < 4:
            base_similarity *= 0.85  # Reduce confidence with limited data
        elif available_features >= 7:
            base_similarity *= 1.05  # Boost confidence with rich data
        
        # Consistency bonus - if multiple color matches are consistent
        color_attrs = ['hair_color', 'shirt_color', 'pants_color', 'shoe_color']
        color_matches = 0
        color_comparisons = 0
        
        for attr in color_attrs:
            val_a = getattr(a, attr, None)
            val_b = getattr(b, attr, None)
            if val_a and val_b and val_a != "unknown" and val_b != "unknown":
                color_comparisons += 1
                if robust_color_match(val_a, val_b) > 0.7:
                    color_matches += 1
        
        if color_comparisons >= 3 and color_matches >= 2:
            base_similarity *= 1.08  # Consistency bonus
        
        return min(1.0, max(0.0, base_similarity))
    
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