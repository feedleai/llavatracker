# Hybrid Person Re-Identification System

A robust person re-identification system that combines YOLOv8 detection, ByteTrack tracking, and multimodal feature extraction (CLIP, face embeddings, and LLaVA appearance descriptions) for persistent person identification across video frames.

## Features

- **YOLOv8 Person Detection**: Fast and accurate person detection using YOLOv8
- **ByteTrack Tracking**: State-of-the-art multi-object tracking for person association
- **Multimodal Feature Extraction**:
  - Face embeddings using InsightFace for robust face recognition
  - Detailed appearance descriptions via LLaVA (hair color, clothing colors/types, etc.)
- **80% Similarity Threshold**: Robust re-identification with 80% feature matching
- **Persistent SQLite Database**: Stores person profiles for long-term re-identification
- **Smart Feature Extraction**: Only processes new persons to optimize performance
- **Clean Visualization**: Shows only global person IDs above heads (no bounding boxes)

## Pipeline Overview

1. **Detection & Tracking**: YOLOv8 detects persons → ByteTrack assigns tracking IDs
2. **Feature Extraction**: For new persons only:
   - Extract detailed appearance (hair/shirt/pants/shoe colors and types)
   - Extract face embeddings if face is visible
3. **Database Storage**: Save person profiles with features in SQLite database
4. **Re-Identification**: When tracking is lost, compare against ALL stored profiles
5. **ID Assignment**: Assign same global ID if 80%+ feature similarity found
6. **Visualization**: Display global person IDs (P1, P2, etc.) above heads

## Installation

```bash
# Clone the repository
git clone https://github.com/feedleai/llavatracker
cd hybrid_reid

# Install dependencies
pip install -r requirements.txt

# Set up LLaVA API key (if using LLaVA features)
export LLAVA_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```bash
python -m hybrid_reid.main --video_path path/to/video.mp4 --output path/to/output.mp4
```

### With Custom Configuration

```bash
python -m hybrid_reid.main \
    --video_path path/to/video.mp4 \
    --config config/custom.yaml \
    --output path/to/output.mp4
```

### Configuration Options

Key configuration parameters in `config/default.yaml`:

```yaml
# Re-identification Settings
reid:
  feature_matching:
    min_similarity: 0.8  # 80% similarity threshold
  database:
    db_path: "reid_profiles.db"  # SQLite database path

# Tracking Settings  
tracker:
  byte_track:
    model_path: "yolov8n.pt"  # YOLOv8 model
    track_thresh: 0.5

# Visualization (clean display)
visualization:
  show_track_ids: false
  show_global_ids: true
  show_confidence: false
```

## Key Components

### 1. Enhanced LLaVA Extractor
Extracts detailed appearance features:
- Hair color and style
- Shirt/pants/shoe colors and types
- Accessories and dominant colors

### 2. SQLite Database
Persistent storage with tables for:
- Person profiles
- CLIP embeddings
- Face embeddings  
- Appearance descriptions

### 3. Smart Re-Identification
- Compares against all stored profiles
- 80% similarity threshold for robust matching
- Handles occlusions and tracking losses

### 4. Clean Visualization
- No bounding boxes
- Only global IDs displayed above heads
- Color-coded person identifiers

## Database Schema

The SQLite database contains:
- `person_profiles`: Main profile data with global IDs
- `face_embeddings`: Face recognition embeddings
- `appearances`: Detailed color and style descriptions

## Performance Optimization

- Feature extraction only for new persons
- Time-decay weighting for historical features
- Efficient SQLite indexing
- Configurable cleanup intervals

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- LLaVA API access (for appearance descriptions)

## Example Output

```
2024-01-01 12:00:00 | INFO | New person detected with track ID 1
2024-01-01 12:00:00 | INFO | New person detected: assigned global ID P1 to track 1
2024-01-01 12:00:05 | INFO | Re-identified person: track 15 matched to global P1 with 85.2% similarity
```

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here] 