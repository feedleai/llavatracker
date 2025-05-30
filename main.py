"""
Main entry point for the hybrid person re-identification system.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import yaml
import cv2
import numpy as np
from loguru import logger

from .tracker.bytetrack_wrapper import ByteTrackWrapper
from .features.base import FeatureExtractionPipeline
from .features.face_embedder import InsightFaceEmbedder
from .features.llava_extractor import LLaVAExtractor
from .reid.feature_database import TimeDecayFeatureDatabase
from .reid.sqlite_database import SQLiteFeatureDatabase
from .reid.id_resolver import IDResolver
from .utils.image_cropper import ImageCropper
from .reid_types import TrackingResult, ReIDResult

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
    config: dict
) -> np.ndarray:
    """
    Visualize tracking and re-identification results on frame.
    Shows only global IDs above people's heads without bounding boxes.
    
    Args:
        frame: Input frame
        result: Re-identification results
        config: Visualization configuration
        
    Returns:
        Annotated frame
    """
    vis_config = config["visualization"]
    colors = vis_config["colors"]
    
    # Create a copy of the frame for visualization
    vis_frame = frame.copy()
    
    # Draw only IDs above heads for each tracked person
    for track_id, global_id in result.assignments.items():
        # Find the tracked person
        person = next(
            (p for p in result.tracked_persons if p.track_id == track_id),
            None
        )
        if person is None:
            continue
        
        # Get color based on global ID
        color = colors[global_id % len(colors)]
        
        # Calculate position above the person's head
        x1, y1, x2, y2 = map(int, person.bbox)
        center_x = (x1 + x2) // 2
        head_y = max(y1 - 20, 20)  # Position above head, with minimum margin from top
        
        # Prepare text - only show global ID
        text = f"P{global_id}"
        
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
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            timestamp = datetime.now()
            
            # Track persons using YOLO + ByteTrack
            tracking_result = tracker.update(frame, frame_id)
            
            # Identify new persons and extract features only for them
            new_persons = []
            for person in tracking_result.tracked_persons:
                if person.track_id not in seen_track_ids:
                    # New person detected - extract features
                    logger.info(f"New person detected with track ID {person.track_id}")
                    feature_pipeline.extract_features(frame, person)
                    new_persons.append(person)
                    seen_track_ids.add(person.track_id)
            
            # Resolve IDs for all tracked persons
            reid_result = id_resolver.resolve_ids(
                tracking_result.tracked_persons,
                feature_db,
                frame_id,
                timestamp
            )
            
            # Update the ReIDResult frame for visualization
            reid_result.frame = frame
            
            # Visualize results (only showing global IDs above heads)
            vis_frame = visualize_frame(frame, reid_result, config)
            
            # Write frame if output is enabled
            if writer is not None:
                writer.write(vis_frame)
            
            # Display frame
            cv2.imshow("Person Re-ID", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            # Cleanup old profiles periodically
            if timestamp - last_cleanup > cleanup_interval:
                feature_db.cleanup_old_profiles(cleanup_interval)
                last_cleanup = timestamp
            
            frame_id += 1
            
            # Log progress
            if frame_id % config["logging"]["log_interval"] == 0:
                logger.info(
                    f"Processed frame {frame_id} | "
                    f"Tracked: {len(tracking_result.tracked_persons)} | "
                    f"New persons: {len(new_persons)} | "
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