#!/usr/bin/env python3
"""
Example usage of the Hybrid Person Re-Identification System.

This script demonstrates how to:
1. Set up the system with custom configuration
2. Process a video for person re-identification
3. Monitor the re-identification results

Run this script with:
    python example_usage.py --video_path path/to/your/video.mp4
"""

import argparse
import os
from pathlib import Path

from hybrid_reid.main import load_config, setup_logging, process_video

def main():
    parser = argparse.ArgumentParser(
        description="Example Person Re-Identification Pipeline"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hybrid_reid/config/default.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set up LLaVA API key if not already set
    if not os.getenv("LLAVA_API_KEY"):
        print("Warning: LLAVA_API_KEY not set. LLaVA features will be disabled.")
        print("To enable LLaVA features, set: export LLAVA_API_KEY='your-api-key'")
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Update database path to output directory
    config["reid"]["database"]["db_path"] = str(output_dir / "reid_profiles.db")
    
    # Update log directory
    config["logging"]["output_dir"] = str(output_dir / "logs")
    Path(config["logging"]["output_dir"]).mkdir(exist_ok=True)
    
    # Setup logging
    setup_logging(config)
    
    # Define output video path
    video_name = Path(args.video_path).stem
    output_video = str(output_dir / f"{video_name}_reid.mp4")
    
    print(f"Processing video: {args.video_path}")
    print(f"Output video: {output_video}")
    print(f"Database: {config['reid']['database']['db_path']}")
    print(f"Configuration:")
    print(f"  - Similarity threshold: {config['reid']['feature_matching']['min_similarity']}")
    print(f"  - Face recognition enabled: {config['features']['face']['enabled']}")
    print(f"  - LLaVA enabled: {config['features']['llava']['enabled']}")
    print("\nStarting processing...")
    print("Press 'q' in the video window to quit early.")
    
    # Process the video
    process_video(
        video_path=args.video_path,
        config=config,
        output_path=output_video
    )
    
    print(f"\nProcessing completed!")
    print(f"Results saved to: {output_dir}")
    print(f"  - Output video: {output_video}")
    print(f"  - Database: {config['reid']['database']['db_path']}")
    print(f"  - Logs: {config['logging']['output_dir']}")

if __name__ == "__main__":
    main() 