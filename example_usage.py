#!/usr/bin/env python3
"""
Example usage of the Hybrid Person Re-Identification System with local LLaVA.

This script demonstrates how to:
1. Set up the system with local Ollama LLaVA server
2. Process a video for person re-identification
3. Monitor the re-identification results

Requirements:
1. Install dependencies: pip install -r requirements.txt
2. Start Ollama server: ollama run llava
3. Run this script: python example_usage.py --video_path test2.mp4
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

from .main import load_config, setup_logging, process_video

def check_gpu_availability():
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available(), torch.cuda.device_count()
    except ImportError:
        return False, 0

def check_ollama_server(server_url="http://localhost:11434"):
    """Check if Ollama server is running and get model info."""
    import requests
    try:
        # Check if Ollama server is running by hitting the tags endpoint
        response = requests.get(f"{server_url}/api/tags", timeout=5)
        if response.status_code != 200:
            return False, None
        
        # Get available models
        models_data = response.json()
        models = models_data.get("models", [])
        
        # Check if llava model is available
        llava_models = [m for m in models if "llava" in m.get("name", "").lower()]
        
        return True, llava_models
    except:
        return False, None

def main():
    parser = argparse.ArgumentParser(
        description="Person Re-Identification with Local LLaVA"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file (e.g., test2.mp4)"
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
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--disable_llava",
        action="store_true",
        help="Disable LLaVA features (use only face recognition)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Check GPU availability
    gpu_available, gpu_count = check_gpu_availability()
    if gpu_available:
        print(f"🚀 GPU detected: {gpu_count} CUDA device(s) available")
    else:
        print("⚠️  No CUDA GPU detected - models will run on CPU (slower)")
    
    # Check if LLaVA should be disabled
    if args.disable_llava:
        config["features"]["llava"]["enabled"] = False
        print("LLaVA disabled - using only face recognition")
    
    # Check Ollama server if LLaVA is enabled
    if config["features"]["llava"]["enabled"]:
        server_url = config["features"]["llava"]["server_url"]
        server_running, model_info = check_ollama_server(server_url)
        
        if not server_running:
            print(f"❌ Ollama server not running at {server_url}")
            print("\nTo start the Ollama server, run in another terminal:")
            print("# First, pull the LLaVA model (if not already done):")
            print("ollama pull llava")
            print("\n# Then start the model:")
            print("ollama run llava")
            print("\nOr run with --disable_llava to use only face recognition")
            return
        else:
            print(f"✅ Ollama server is running at {server_url}")
            
            if model_info:
                if len(model_info) > 0:
                    print(f"   📋 LLaVA model available: {model_info[0].get('name', 'Unknown')}")
                else:
                    print("   ⚠️  No LLaVA model found. Run 'ollama pull llava' to install it.")
            else:
                print("   ⚠️  Could not verify model availability")
    
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
    
    print(f"\n🎬 Processing video: {args.video_path}")
    print(f"📁 Output video: {output_video}")
    print(f"🗃️  Database: {config['reid']['database']['db_path']}")
    print(f"\nConfiguration:")
    print(f"  - Similarity threshold: {config['reid']['feature_matching']['min_similarity']}")
    print(f"  - Face recognition: {'✅ Enabled' if config['features']['face']['enabled'] else '❌ Disabled'}")
    print(f"  - LLaVA appearance: {'✅ Enabled' if config['features']['llava']['enabled'] else '❌ Disabled'}")
    
    if config['features']['llava']['enabled']:
        print(f"  - LLaVA model: {config['features']['llava']['model_name']}")
    
    print("\n🚀 Starting processing...")
    print("📺 Video preview window will open showing:")
    print("   • Real-time person tracking and re-identification")
    print("   • Global person IDs (P1, P2, etc.) above each person")
    print("   • 2-second wait timer for new persons before feature extraction")
    print("   • Yellow 'Wait X.Xs' labels during the waiting period")
    print("   • Live statistics (FPS, frame count, person count, waiting count)")
    print("   • Press 'q' in the video window to quit early")
    print("\n⏳ Please wait for the preview window to appear...")
    
    # Process the video
    try:
        process_video(
            video_path=args.video_path,
            config=config,
            output_path=output_video
        )
        
        print(f"\n✅ Processing completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"  - Output video: {output_video}")
        print(f"  - Database: {config['reid']['database']['db_path']}")
        print(f"  - Logs: {config['logging']['output_dir']}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing stopped by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")

if __name__ == "__main__":
    main() 