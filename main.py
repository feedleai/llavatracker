"""
Main entry point for the hybrid person re-identification system.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import yaml
import cv2
import numpy as np
from loguru import logger
import time
import json

from .tracker.bytetrack_wrapper import ByteTrackWrapper
from .features.base import FeatureExtractionPipeline
from .features.face_embedder import InsightFaceEmbedder
from .features.llava_extractor import LLaVAExtractor
from .reid.feature_database import TimeDecayFeatureDatabase
from .reid.sqlite_database import SQLiteFeatureDatabase
from .reid.id_resolver import IDResolver
from .utils.image_cropper import ImageCropper
from .reid_types import TrackingResult, ReIDResult, TrackedPerson, PersonProfile, AppearanceDescription

def setup_logging(config: dict) -> None:
    """Configure logging based on config."""
    log_config = config["logging"]
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=log_config["level"],
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>"
    )
    
    # Add file handler if output directory exists
    output_dir = Path(log_config["output_dir"])
    if output_dir.exists():
        logger.add(
            output_dir / "reid_{time}.log",
            rotation="1 day",
            retention="7 days",
            level=log_config["level"]
        )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def visualize_frame(
    frame: np.ndarray,
    result: ReIDResult,
    tracked_persons: List[TrackedPerson],
    config: dict,
    frame_id: int = 0,
    fps: float = 0.0,
    total_persons: int = 0,
    new_persons_count: int = 0,
    person_detection_times: dict = None,  # Add detection times
    persons_with_features: set = None,   # Add feature status (now for face only)
    current_timestamp: datetime = None   # Add current timestamp
) -> np.ndarray:
    """
    Visualize tracking and re-identification results on frame.
    Shows only global IDs above people's heads without bounding boxes,
    plus real-time statistics overlay.
    
    Args:
        frame: Input frame
        result: Re-identification results
        tracked_persons: List of tracked persons in this frame
        config: Visualization configuration
        frame_id: Current frame number
        fps: Current processing FPS
        total_persons: Total unique persons seen so far
        new_persons_count: Number of new persons in this frame
        person_detection_times: Dictionary to track detection times
        persons_with_features: Set to track which persons have had FACE features extracted
        current_timestamp: Current timestamp for comparison
        
    Returns:
        Annotated frame
    """
    vis_config = config["visualization"]
    colors = vis_config["colors"]
    
    # Create a copy of the frame for visualization
    vis_frame = frame.copy()
    
    # Draw IDs and status above heads for each tracked person
    for person in tracked_persons:
        track_id = person.track_id
        
        # Calculate position above the person's head
        x1, y1, x2, y2 = map(int, person.bbox)
        center_x = (x1 + x2) // 2
        head_y = max(y1 - 20, 20)  # Position above head, with minimum margin from top
        
        # Determine what to display based on person's status
        if track_id in result.assignments:
            # Person has been assigned a global ID
            global_id = result.assignments[track_id]
            color = colors[global_id % len(colors)]
            text = f"P{global_id}"
        else:
            # Fallback for unknown state (processing)
            color = (128, 128, 128)  # Gray
            text = f"T{track_id}"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = vis_config.get("text_scale", 0.8)
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw text background for better visibility
        bg_margin = 5
        cv2.rectangle(
            vis_frame,
            (center_x - text_w // 2 - bg_margin, head_y - text_h - bg_margin),
            (center_x + text_w // 2 + bg_margin, head_y + baseline + bg_margin),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            vis_frame,
            text,
            (center_x - text_w // 2, head_y),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )
    
    # Add statistics overlay in top-left corner
    height, width = vis_frame.shape[:2]
    overlay_font = cv2.FONT_HERSHEY_SIMPLEX
    overlay_font_scale = 0.6
    overlay_thickness = 1
    overlay_color = (255, 255, 255)  # White text
    bg_color = (0, 0, 0)  # Black background
    
    # Prepare statistics text
    stats_lines = [
        f"Frame: {frame_id}",
        f"FPS: {fps:.1f}",
        f"Tracked: {len(tracked_persons)}",
        f"Total Persons: {total_persons}",
        f"New This Frame: {new_persons_count}",
        f"ID Assignment: Appearance-first",
        f"Extraction: Immediate (50% body visible)",
        f"Press 'q' to quit"
    ]
    
    # Calculate overlay size
    max_text_width = 0
    line_height = 0
    for line in stats_lines:
        (text_w, text_h), baseline = cv2.getTextSize(line, overlay_font, overlay_font_scale, overlay_thickness)
        max_text_width = max(max_text_width, text_w)
        line_height = max(line_height, text_h + baseline)
    
    # Draw semi-transparent background for statistics
    overlay_padding = 10
    overlay_width = max_text_width + 2 * overlay_padding
    overlay_height = len(stats_lines) * line_height + 2 * overlay_padding
    
    # Create overlay background
    overlay = vis_frame.copy()
    cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
    
    # Draw statistics text
    y_offset = overlay_padding + line_height
    for line in stats_lines:
        cv2.putText(
            vis_frame,
            line,
            (overlay_padding, y_offset),
            overlay_font,
            overlay_font_scale,
            overlay_color,
            overlay_thickness
        )
        y_offset += line_height
    
    return vis_frame

def save_person_features_json(person_features: dict, output_path: str):
    """
    Save person appearance features to JSON file for debugging and monitoring.
    Uses the new structured format with nested objects for cleaner organization.
    
    Args:
        person_features: Dictionary containing all person features
        output_path: Path to save the JSON file
    """
    try:
        logger.debug(f"Attempting to save {len(person_features)} person features to JSON")
        
        # Create clean JSON with new structured appearance attributes
        json_features = {}
        for person_id, features in person_features.items():
            logger.debug(f"Processing {person_id} with features: {list(features.keys())}")
            
            if "appearance_description" in features and features["appearance_description"]:
                appearance = features["appearance_description"]
                logger.debug(f"Found appearance data for {person_id}: {appearance}")
                
                # Extract structured appearance attributes in new format
                json_features[person_id] = {
                    "gender": appearance.get("gender_guess", "unknown"),
                    "age": appearance.get("age_range", "unknown"),
                    "hair": {
                        "color": appearance.get("hair_color", "unknown"),
                        "style": appearance.get("hair_style", "unknown")
                    },
                    "headwear": {
                        "type": appearance.get("headwear_type", "none"),
                        "color": appearance.get("headwear_color", "none")
                    },
                    "facial_accessories": appearance.get("facial_features_accessories", []),
                    "upper_clothing": {
                        "type": appearance.get("upper_clothing_type", "unknown"),
                        "color_primary": appearance.get("upper_clothing_color_primary", "unknown"),
                        "color_secondary": appearance.get("upper_clothing_color_secondary", []),
                        "pattern": appearance.get("upper_clothing_pattern_or_print", "none"),
                        "sleeve_length": appearance.get("sleeve_length", "unknown")
                    },
                    "lower_clothing": {
                        "type": appearance.get("lower_clothing_type", "unknown"),
                        "color": appearance.get("lower_clothing_color", "unknown"),
                        "pattern": appearance.get("lower_clothing_pattern", "none")
                    },
                    "footwear": {
                        "type": appearance.get("footwear_type", "unknown"),
                        "color": appearance.get("footwear_color", "unknown")
                    },
                    "accessories": appearance.get("carried_items_or_prominent_accessories", []),
                    "dominant_colors": appearance.get("dominant_colors_overall_outfit", []),
                    "distinctive_features": appearance.get("other_distinctive_visual_cues", "none")
                }
                logger.debug(f"Added {person_id} to JSON with appearance: {json_features[person_id]}")
            else:
                logger.debug(f"No appearance_description found for {person_id}")
        
        logger.info(f"Saving {len(json_features)} persons with appearance data to {output_path}")
        
        # Save to JSON file with pretty formatting
        with open(output_path, 'w') as f:
            json.dump(json_features, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Person appearance features saved to {output_path} ({len(json_features)} persons)")
    except Exception as e:
        logger.error(f"Failed to save person features to JSON: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def calculate_body_visibility(bbox: tuple, frame_shape: tuple) -> float:
    """
    Calculate what percentage of the body is visible in the frame.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        frame_shape: Frame dimensions (height, width)
        
    Returns:
        Visibility percentage (0.0 to 1.0)
    """
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame_shape[:2]
    
    # Calculate the area within frame bounds
    visible_x1 = max(0, x1)
    visible_y1 = max(0, y1)
    visible_x2 = min(frame_width, x2)
    visible_y2 = min(frame_height, y2)
    
    # Calculate areas
    total_bbox_area = (x2 - x1) * (y2 - y1)
    visible_area = max(0, (visible_x2 - visible_x1) * (visible_y2 - visible_y1))
    
    if total_bbox_area <= 0:
        return 0.0
    
    return visible_area / total_bbox_area

def process_video(
    video_path: str,
    config: dict,
    output_path: Optional[str] = None
) -> None:
    """
    Process a video file for person re-identification.
    
    Args:
        video_path: Path to input video file
        config: Configuration dictionary
        output_path: Optional path to save output video
    """
    # Initialize components
    tracker = ByteTrackWrapper(config["tracker"]["byte_track"])
    
    # Initialize feature extractors
    face_extractor = InsightFaceEmbedder() if config["features"]["face"]["enabled"] else None
    llava_extractor = LLaVAExtractor() if config["features"]["llava"]["enabled"] else None
    
    feature_pipeline = FeatureExtractionPipeline(
        face_extractor=face_extractor,
        appearance_extractor=llava_extractor,
        config=config["features"]
    )
    
    # Initialize re-identification components with SQLite database
    feature_db = SQLiteFeatureDatabase(config["reid"])
    id_resolver = IDResolver(config["reid"])
    
    # Initialize person features tracking for JSON export
    person_features = {}  # Will store: {person_id: {feature_type: data}}
    json_output_path = Path(video_path).parent / "person_features.json"
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )
    
    frame_id = 0
    last_cleanup = datetime.now()
    cleanup_interval = timedelta(seconds=config["reid"]["database"]["cleanup_interval"])
    
    # Track which persons we've seen to only extract features for new ones
    seen_track_ids = set()
    total_unique_persons = 0
    
    # Track when persons were first detected (no delays anymore)
    person_detection_times = {}  # track_id -> first_detection_timestamp
    persons_with_face_features = set()  # track_ids that have had FACE features extracted
    persons_with_appearance_features = set()  # track_ids that have had APPEARANCE features extracted
    
    # New: Track ID to Global ID mapping (this is the key change)
    track_to_global_mapping = {}  # track_id -> global_id
    next_global_id = 1  # Counter for assigning new global IDs
    
    # For FPS calculation
    frame_start_time = time.time()
    fps_history = []
    
    try:
        # Create named window with resizable option
        cv2.namedWindow("Person Re-ID Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Person Re-ID Preview", 1280, 720)  # Default size
        
        print("\n🎬 Video preview window opened!")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Resize window as needed")
        print("  - Global IDs are shown above each person's head")
        print(f"  - Person features will be saved to: {json_output_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            timestamp = datetime.now()
            
            # Track persons using YOLO + ByteTrack
            tracking_result = tracker.update(frame, frame_id)
            
            # Identify new persons and persons ready for feature extraction
            new_persons = []
            persons_ready_for_face_features = []
            persons_ready_for_appearance = []
            
            for person in tracking_result.tracked_persons:
                # Track when this person was first detected
                if person.track_id not in seen_track_ids:
                    person_detection_times[person.track_id] = timestamp
                    new_persons.append(person)
                    seen_track_ids.add(person.track_id)
                    logger.info(f"New track ID {person.track_id} detected")
                
                # Check if this person is ready for FACE feature extraction (immediate, once only)
                if person.track_id not in persons_with_face_features:
                    persons_ready_for_face_features.append(person)
                    persons_with_face_features.add(person.track_id)
                    logger.info(f"Track ID {person.track_id} ready for immediate face extraction")
                
                # Check if this person is ready for APPEARANCE extraction (immediate, once only, with body visibility check)
                if person.track_id not in persons_with_appearance_features:
                    # Check body visibility (at least 50% visible)
                    body_visibility = calculate_body_visibility(person.bbox, frame.shape)
                    if body_visibility >= 0.5:
                        persons_ready_for_appearance.append(person)
                        persons_with_appearance_features.add(person.track_id)
                        logger.info(f"Track ID {person.track_id} ready for appearance extraction ({body_visibility:.1%} visible)")
                    else:
                        logger.debug(f"Track ID {person.track_id} body visibility too low ({body_visibility:.1%}) - skipping appearance extraction")
            
            # STEP 1: Extract APPEARANCE features FIRST for persons ready (appearance-first ID assignment)
            for person in persons_ready_for_appearance:
                logger.info(f"Extracting appearance features for track ID {person.track_id} (appearance-first)")
                
                try:
                    # Extract only appearance features
                    updated_person = feature_pipeline.extract_features(frame, person)
                    
                    appearance_description = None
                    if updated_person and hasattr(updated_person, 'appearance') and updated_person.appearance:
                        appearance_description = updated_person.appearance
                        logger.info(f"Appearance description extracted for track {person.track_id}")
                    
                    # APPEARANCE-FIRST MATCHING: Check database for appearance matches
                    if appearance_description is not None:
                        matched_global_id = None
                        best_similarity = 0.0
                        
                        existing_profiles = feature_db.get_all_profiles()
                        for profile in existing_profiles:
                            if profile.appearances:
                                for timestamp_app, stored_appearance in profile.appearances:
                                    appearance_similarity = feature_db._enhanced_appearance_similarity(appearance_description, stored_appearance)
                                    if appearance_similarity > 0.8 and appearance_similarity > best_similarity:
                                        best_similarity = appearance_similarity
                                        matched_global_id = profile.global_id
                                        logger.info(f"Appearance match found for track {person.track_id}: {appearance_similarity:.3f} with person {profile.global_id}")
                        
                        if matched_global_id is not None:
                            # Found appearance match - assign to existing person
                            track_to_global_mapping[person.track_id] = matched_global_id
                            logger.info(f"Track {person.track_id} assigned to existing person {matched_global_id} via appearance match")
                            
                            # Update existing profile
                            existing_profile = feature_db.get_profile(matched_global_id)
                            if existing_profile:
                                existing_profile.add_features(timestamp, None, appearance_description)
                                existing_profile.track_ids.append(person.track_id)
                                existing_profile.last_seen = timestamp
                                feature_db.update_profile(existing_profile)
                        else:
                            # No appearance match - create new person with new ID
                            new_global_id = next_global_id
                            next_global_id += 1
                            total_unique_persons += 1
                            track_to_global_mapping[person.track_id] = new_global_id
                            logger.info(f"Track {person.track_id} assigned new global ID {new_global_id} (new appearance)")
                            
                            # Create new profile
                            new_profile = PersonProfile(
                                global_id=new_global_id,
                                first_seen=timestamp,
                                last_seen=timestamp,
                                track_ids=[person.track_id]
                            )
                            new_profile.add_features(timestamp, None, appearance_description)
                            feature_db.add_profile(new_profile)
                    else:
                        # No appearance extracted - create new person anyway
                        new_global_id = next_global_id
                        next_global_id += 1
                        total_unique_persons += 1
                        track_to_global_mapping[person.track_id] = new_global_id
                        logger.info(f"Track {person.track_id} assigned new global ID {new_global_id} (no appearance)")
                        
                        # Create basic profile
                        new_profile = PersonProfile(
                            global_id=new_global_id,
                            first_seen=timestamp,
                            last_seen=timestamp,
                            track_ids=[person.track_id]
                        )
                        feature_db.add_profile(new_profile)
                    
                    # Store features for JSON export
                    if person.track_id in track_to_global_mapping:
                        global_id = track_to_global_mapping[person.track_id]
                        track_features = {
                            "track_id": person.track_id,
                            "extraction_timestamp": timestamp.isoformat(),
                            "frame_id": frame_id,
                            "appearance_description": None
                        }
                        
                        if appearance_description:
                            if hasattr(appearance_description, 'dict'):
                                track_features["appearance_description"] = appearance_description.dict()
                            elif hasattr(appearance_description, '__dict__'):
                                track_features["appearance_description"] = appearance_description.__dict__
                            else:
                                # Map new detailed fields for JSON export
                                track_features["appearance_description"] = {
                                    "gender_guess": getattr(appearance_description, 'gender_guess', 'unknown'),
                                    "age_range": getattr(appearance_description, 'age_range', 'unknown'),
                                    "hair_color": getattr(appearance_description, 'hair_color', 'unknown'),
                                    "hair_style": getattr(appearance_description, 'hair_style', 'unknown'),
                                    "headwear_type": getattr(appearance_description, 'headwear_type', 'unknown'),
                                    "headwear_color": getattr(appearance_description, 'headwear_color', 'unknown'),
                                    "facial_features_accessories": getattr(appearance_description, 'facial_features_accessories', []),
                                    "upper_clothing_color_primary": getattr(appearance_description, 'upper_clothing_color_primary', 'unknown'),
                                    "upper_clothing_color_secondary": getattr(appearance_description, 'upper_clothing_color_secondary', []),
                                    "upper_clothing_type": getattr(appearance_description, 'upper_clothing_type', 'unknown'),
                                    "upper_clothing_pattern_or_print": getattr(appearance_description, 'upper_clothing_pattern_or_print', 'none'),
                                    "sleeve_length": getattr(appearance_description, 'sleeve_length', 'unknown'),
                                    "lower_clothing_color": getattr(appearance_description, 'lower_clothing_color', 'unknown'),
                                    "lower_clothing_type": getattr(appearance_description, 'lower_clothing_type', 'unknown'),
                                    "lower_clothing_pattern": getattr(appearance_description, 'lower_clothing_pattern', 'none'),
                                    "footwear_color": getattr(appearance_description, 'footwear_color', 'unknown'),
                                    "footwear_type": getattr(appearance_description, 'footwear_type', 'unknown'),
                                    "carried_items_or_prominent_accessories": getattr(appearance_description, 'carried_items_or_prominent_accessories', []),
                                    "dominant_colors_overall_outfit": getattr(appearance_description, 'dominant_colors_overall_outfit', []),
                                    "other_distinctive_visual_cues": getattr(appearance_description, 'other_distinctive_visual_cues', 'none')
                                }
                        
                        person_features[f"person_{global_id}"] = track_features
                
                except Exception as e:
                    logger.error(f"Appearance feature extraction failed for track {person.track_id}: {e}")
                    # Still assign an ID even if extraction fails
                    if person.track_id not in track_to_global_mapping:
                        new_global_id = next_global_id
                        next_global_id += 1
                        total_unique_persons += 1
                        track_to_global_mapping[person.track_id] = new_global_id
                        
                        # Create basic profile
                        new_profile = PersonProfile(
                            global_id=new_global_id,
                            first_seen=timestamp,
                            last_seen=timestamp,
                            track_ids=[person.track_id]
                        )
                        feature_db.add_profile(new_profile)
            
            # STEP 2: Extract FACE features for persons ready (immediate, once only) and add to existing profiles
            for person in persons_ready_for_face_features:
                # Only extract face features if this person already has a global ID (from appearance step)
                if person.track_id in track_to_global_mapping:
                    logger.info(f"Extracting face features for track ID {person.track_id} (adding to existing profile)")
                    
                    try:
                        # Extract only face features
                        updated_person = feature_pipeline.extract_features(frame, person)
                        
                        face_embedding = None
                        if updated_person and hasattr(updated_person, 'face_embedding') and updated_person.face_embedding is not None:
                            face_embedding = updated_person.face_embedding
                            logger.info(f"Face embedding extracted for track {person.track_id}")
                        
                        # Add face features to existing person profile
                        if face_embedding is not None:
                            existing_global_id = track_to_global_mapping[person.track_id]
                            existing_profile = feature_db.get_profile(existing_global_id)
                            
                            if existing_profile:
                                logger.info(f"Adding face embedding to existing person {existing_global_id} (track {person.track_id})")
                                existing_profile.add_features(timestamp, face_embedding, None)
                                feature_db.update_profile(existing_profile)
                    
                    except Exception as e:
                        logger.error(f"Face feature extraction failed for track {person.track_id}: {e}")
                else:
                    logger.debug(f"Skipping face extraction for track {person.track_id} - no global ID assigned yet")
            
            # Create reid result using our track_to_global_mapping (no more complex ID resolution)
            reid_result_assignments = {}
            reid_new_ids = []
            reid_reused_ids = []
            
            for person in tracking_result.tracked_persons:
                if person.track_id in track_to_global_mapping:
                    global_id = track_to_global_mapping[person.track_id]
                    reid_result_assignments[person.track_id] = global_id
                    
                    # Check if this is a new assignment from this frame
                    if person.track_id in persons_ready_for_face_features or person.track_id in persons_ready_for_appearance:
                        # Check if this global_id was newly created or reused
                        if any(pid for pid in person_features.keys() if pid == f"person_{global_id}"):
                            # This global_id already existed, so it's reused
                            if global_id not in reid_reused_ids:
                                reid_reused_ids.append(global_id)
                        else:
                            # This is a new global_id
                            if global_id not in reid_new_ids:
                                reid_new_ids.append(global_id)
            
            from .reid_types import ReIDResult
            reid_result = ReIDResult(
                frame_id=frame_id,
                timestamp=timestamp,
                assignments=reid_result_assignments,
                new_global_ids=reid_new_ids,
                reused_global_ids=reid_reused_ids,
                frame=frame
            )
            
            # Save person features to JSON periodically (every 30 frames)
            if frame_id % 30 == 0 and person_features:
                save_person_features_json(person_features, str(json_output_path))
            
            # Calculate FPS
            frame_end_time = time.time()
            frame_fps = 1.0 / (frame_end_time - frame_start_time) if frame_end_time > frame_start_time else 0
            fps_history.append(frame_fps)
            if len(fps_history) > 30:  # Keep last 30 frames for smoothing
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            frame_start_time = frame_end_time
            
            # Visualize results with enhanced overlay
            vis_frame = visualize_frame(
                frame, 
                reid_result, 
                tracking_result.tracked_persons,
                config,
                frame_id=frame_id,
                fps=avg_fps,
                total_persons=total_unique_persons,
                new_persons_count=len(new_persons),
                person_detection_times=person_detection_times,
                persons_with_features=persons_with_face_features,
                current_timestamp=timestamp
            )
            
            # Write frame if output is enabled
            if writer is not None:
                writer.write(vis_frame)
            
            # Display frame with enhanced window
            cv2.imshow("Person Re-ID Preview", vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n⏹️  Video preview stopped by user")
                break
            
            # Cleanup old profiles periodically
            if timestamp - last_cleanup > cleanup_interval:
                feature_db.cleanup_old_profiles(cleanup_interval)
                
                # Also cleanup old detection times for tracks no longer active
                current_track_ids = {p.track_id for p in tracking_result.tracked_persons}
                old_detection_times = [
                    track_id for track_id in person_detection_times.keys()
                    if track_id not in current_track_ids
                ]
                for track_id in old_detection_times:
                    del person_detection_times[track_id]
                    persons_with_face_features.discard(track_id)
                    # Clean up appearance extraction tracking too
                    persons_with_appearance_features.discard(track_id)
                
                last_cleanup = timestamp
            
            frame_id += 1
            
            # Log progress
            if frame_id % config["logging"]["log_interval"] == 0:
                logger.info(
                    f"Processed frame {frame_id} | "
                    f"FPS: {avg_fps:.1f} | "
                    f"Tracked: {len(tracking_result.tracked_persons)} | "
                    f"Appearance extracted: {len(persons_ready_for_appearance)} | "
                    f"Face extracted: {len(persons_ready_for_face_features)} | "
                    f"New persons: {len(new_persons)} | "
                    f"Total unique: {total_unique_persons} | "
                    f"New IDs: {len(reid_result.new_global_ids)} | "
                    f"Reused IDs: {len(reid_result.reused_global_ids)} | "
                    f"JSON entries: {len(person_features)}"
                )
    
    finally:
        # Save final person features to JSON
        if person_features:
            save_person_features_json(person_features, str(json_output_path))
            logger.info(f"Final person features saved to {json_output_path}")
        
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        feature_pipeline.cleanup()
        feature_db.close()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid Person Re-Identification System"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output video"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Process video
    logger.info(f"Starting video processing: {args.video_path}")
    process_video(args.video_path, config, args.output)
    logger.info("Video processing completed")

if __name__ == "__main__":
    main() 