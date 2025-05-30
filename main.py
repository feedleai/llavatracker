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

from .tracker.bytetrack_wrapper import ByteTrackWrapper
from .features.base import FeatureExtractionPipeline
from .features.face_embedder import InsightFaceEmbedder
from .features.llava_extractor import LLaVAExtractor
from .reid.feature_database import TimeDecayFeatureDatabase
from .reid.sqlite_database import SQLiteFeatureDatabase
from .reid.id_resolver import IDResolver
from .utils.image_cropper import ImageCropper
from .reid_types import TrackingResult, ReIDResult, TrackedPerson

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
    persons_with_features: set = None,   # Add feature status
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
        persons_with_features: Set to track which persons have had features extracted
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
        elif (person_detection_times and persons_with_features and 
              track_id in person_detection_times and 
              track_id not in persons_with_features):
            # Person is waiting for feature extraction
            if current_timestamp:
                time_waiting = current_timestamp - person_detection_times[track_id]
                remaining_time = max(0, 2.0 - time_waiting.total_seconds())
                color = (0, 255, 255)  # Yellow for waiting
                text = f"Wait {remaining_time:.1f}s"
            else:
                color = (0, 255, 255)  # Yellow for waiting
                text = "Waiting..."
        else:
            # Fallback for unknown state
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
    waiting_count = 0
    if person_detection_times and persons_with_features:
        waiting_count = len([
            track_id for track_id in person_detection_times.keys()
            if track_id not in persons_with_features
        ])
    
    stats_lines = [
        f"Frame: {frame_id}",
        f"FPS: {fps:.1f}",
        f"Tracked: {len(tracked_persons)}",
        f"Waiting: {waiting_count}",
        f"Total Persons: {total_persons}",
        f"New This Frame: {new_persons_count}",
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
    
    # Track when persons were first detected (for 2-second delay)
    person_detection_times = {}  # track_id -> first_detection_timestamp
    persons_with_features = set()  # track_ids that have had features extracted
    feature_extraction_delay = timedelta(seconds=2)  # 2-second delay
    
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
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            timestamp = datetime.now()
            
            # Track persons using YOLO + ByteTrack
            tracking_result = tracker.update(frame, frame_id)
            
            # Identify new persons and track detection times
            new_persons = []
            persons_ready_for_features = []
            
            for person in tracking_result.tracked_persons:
                # Track when this person was first detected
                if person.track_id not in seen_track_ids:
                    person_detection_times[person.track_id] = timestamp
                    new_persons.append(person)
                    seen_track_ids.add(person.track_id)
                    total_unique_persons += 1
                    logger.info(f"New person detected with track ID {person.track_id}, waiting 2 seconds for feature extraction")
                
                # Check if this person is ready for feature extraction (2 seconds have passed)
                if (person.track_id not in persons_with_features and 
                    person.track_id in person_detection_times):
                    time_since_detection = timestamp - person_detection_times[person.track_id]
                    if time_since_detection >= feature_extraction_delay:
                        persons_ready_for_features.append(person)
                        persons_with_features.add(person.track_id)
                        logger.info(f"Person {person.track_id} ready for feature extraction after {time_since_detection.total_seconds():.1f} seconds")
            
            # Extract features only for persons who have waited 2 seconds
            for person in persons_ready_for_features:
                logger.info(f"Extracting features for person with track ID {person.track_id}")
                feature_pipeline.extract_features(frame, person)
            
            # Resolve IDs for all tracked persons
            reid_result = id_resolver.resolve_ids(
                tracking_result.tracked_persons,
                feature_db,
                frame_id,
                timestamp,
                frame
            )
            
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
                persons_with_features=persons_with_features,
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
                    persons_with_features.discard(track_id)
                
                last_cleanup = timestamp
            
            frame_id += 1
            
            # Log progress
            if frame_id % config["logging"]["log_interval"] == 0:
                waiting_count = len([
                    track_id for track_id in person_detection_times.keys()
                    if track_id not in persons_with_features
                ])
                logger.info(
                    f"Processed frame {frame_id} | "
                    f"FPS: {avg_fps:.1f} | "
                    f"Tracked: {len(tracking_result.tracked_persons)} | "
                    f"Waiting: {waiting_count} | "
                    f"Features extracted: {len(persons_ready_for_features)} | "
                    f"New persons: {len(new_persons)} | "
                    f"Total unique: {total_unique_persons} | "
                    f"New IDs: {len(reid_result.new_global_ids)} | "
                    f"Reused IDs: {len(reid_result.reused_global_ids)}"
                )
    
    finally:
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