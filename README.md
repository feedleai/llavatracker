# Hybrid Person Re-Identification System

A robust person re-identification system that combines YOLOv8 detection, ByteTrack tracking, and multimodal feature extraction (face embeddings and LLaVA appearance descriptions) for persistent person identification across video frames.

## Features

- **YOLOv8 Person Detection**: Fast and accurate person detection using YOLOv8
- **ByteTrack Tracking**: State-of-the-art multi-object tracking for person association
- **Multimodal Feature Extraction**:
  - Face embeddings using InsightFace for robust face recognition
  - Detailed appearance descriptions via local LLaVA model (hair color, clothing colors/types, etc.)
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

## Installation & Setup

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd hybrid_reid

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Local LLaVA Server

The system uses a local LLaVA model served via vLLM. Start the server in a separate terminal:

```bash
# Install vLLM (if not already installed)
pip install vllm

# Start the LLaVA server (this will download the model on first run)
vllm serve liuhaotian/llava-v1.5-7b
```

**Note**: The first run will download the LLaVA-1.5-7B model (~13GB). Subsequent runs will use the cached model.

### 3. Verify Server is Running

The server should start at `http://localhost:8000`. You can test it with:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "liuhaotian/llava-v1.5-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Usage

### Basic Usage with test2.mp4

```bash
# Run with LLaVA enabled (server must be running)
python example_usage.py --video_path test2.mp4

# Run without LLaVA (face recognition only)
python example_usage.py --video_path test2.mp4 --disable_llava
```

### Advanced Usage

```bash
# Custom output directory
python example_usage.py --video_path test2.mp4 --output_dir ./my_results

# Custom configuration
python example_usage.py --video_path test2.mp4 --config my_config.yaml

# Direct main module usage
python -m hybrid_reid.main --video_path test2.mp4 --output output.mp4
```

## Configuration

Key settings in `config/default.yaml`:

```yaml
# Re-identification Settings
reid:
  feature_matching:
    min_similarity: 0.8  # 80% similarity threshold
  database:
    db_path: "reid_profiles.db"

# Local LLaVA Settings
features:
  llava:
    enabled: true
    server_url: "http://localhost:8000"
    model_name: "liuhaotian/llava-v1.5-7b"
  face:
    enabled: true
    similarity_threshold: 0.6

# Tracking Settings  
tracker:
  byte_track:
    model_path: "yolov8n.pt"  # Will download automatically
    track_thresh: 0.5
```

## System Requirements

- **Python**: 3.8+
- **GPU**: CUDA-capable GPU recommended (for LLaVA and YOLOv8)
- **RAM**: 16GB+ recommended (LLaVA model uses ~8GB VRAM)
- **Disk**: ~15GB for models and cache

## Key Components

### 1. Local LLaVA Model
- Runs via vLLM server on localhost:8000
- Extracts detailed appearance features:
  - Hair color and style
  - Shirt/pants/shoe colors and types
  - Accessories and dominant colors

### 2. SQLite Database
Persistent storage with tables for:
- Person profiles
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

## Example Output

```bash
✅ vLLM server is running at http://localhost:8000

🎬 Processing video: test2.mp4
📁 Output video: ./output/test2_reid.mp4
🗃️ Database: ./output/reid_profiles.db

Configuration:
  - Similarity threshold: 0.8
  - Face recognition: ✅ Enabled
  - LLaVA appearance: ✅ Enabled
  - LLaVA model: liuhaotian/llava-v1.5-7b

🚀 Starting processing...
```

## Troubleshooting

### vLLM Server Issues

```bash
# Check if server is running
curl http://localhost:8000/health

# Start server if not running
vllm serve liuhaotian/llava-v1.5-7b

# Use different port if 8000 is busy
vllm serve liuhaotian/llava-v1.5-7b --port 8001
```

### Memory Issues

If you encounter CUDA out of memory errors:

```bash
# Use smaller model or reduce batch size
vllm serve liuhaotian/llava-v1.5-7b --gpu-memory-utilization 0.8

# Or disable LLaVA and use face recognition only
python example_usage.py --video_path test2.mp4 --disable_llava
```

## Performance Optimization

- Feature extraction only for new persons
- Time-decay weighting for historical features
- Efficient SQLite indexing
- Configurable cleanup intervals

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here] 