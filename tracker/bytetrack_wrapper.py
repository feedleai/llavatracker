"""
YOLOv8 person detection with ByteTrack tracking wrapper.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np
from datetime import datetime
from ultralytics import YOLO

from ..types import (
    BoundingBox,
    TrackID,
    FrameID,
    Confidence,
    TrackedPerson,
    TrackingResult
)

class BaseTracker(ABC):
    """Abstract base class for trackers."""
    
    @abstractmethod
    def initialize(self, config: dict) -> None:
        """Initialize the tracker with configuration."""
        pass
    
    @abstractmethod
    def update(self, frame: np.ndarray, frame_id: FrameID) -> TrackingResult:
        """Update tracking with a new frame."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the tracker state."""
        pass

class ByteTrackWrapper(BaseTracker):
    """YOLOv8 + ByteTrack wrapper for person detection and tracking."""
    
    def __init__(self, config: dict):
        """
        Initialize ByteTrack wrapper with YOLOv8.
        
        Args:
            config: Dictionary containing tracking configuration parameters
        """
        self.config = config
        self.model = None
        self.frame_id = 0
        self.initialize(config)
    
    def initialize(self, config: dict) -> None:
        """
        Initialize YOLOv8 with ByteTrack tracker.
        
        Args:
            config: Dictionary containing parameters:
                - model_path: Path to YOLO model (default: 'yolov8n.pt')
                - conf_threshold: Detection confidence threshold
                - iou_threshold: NMS IOU threshold
                - device: Device to run on ('cpu', 'cuda', etc.)
                - track_thresh: ByteTrack track threshold
                - track_buffer: ByteTrack track buffer
                - match_thresh: ByteTrack match threshold
        """
        model_path = config.get('model_path', 'yolov8n.pt')
        self.conf_threshold = config.get('track_thresh', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.7)
        self.device = config.get('device', 'cuda' if self._cuda_available() else 'cpu')
        
        # Initialize YOLOv8 model
        self.model = YOLO(model_path)
        
        # Configure the model for tracking
        self.model.to(self.device)
    
    def update(self, frame: np.ndarray, frame_id: FrameID) -> TrackingResult:
        """
        Update tracking with a new frame using YOLOv8 + ByteTrack.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            frame_id: Frame identifier
            
        Returns:
            TrackingResult containing tracked persons and frame
        """
        timestamp = datetime.now()
        tracked_persons = []
        
        if self.model is not None:
            # Run YOLOv8 detection and tracking with ByteTrack
            results = self.model.track(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                classes=[0],  # Only detect persons (class 0 in COCO)
                tracker="bytetrack.yaml",  # Use ByteTrack
                persist=True,
                verbose=False
            )
            
            # Extract tracking results
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    
                    for bbox, conf, track_id in zip(boxes, confidences, track_ids):
                        # Filter minimum box area if specified
                        min_area = self.config.get('min_box_area', 100)
                        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        
                        if bbox_area >= min_area:
                            tracked_person = TrackedPerson(
                                track_id=int(track_id),
                                bbox=self._convert_bbox(bbox),
                                confidence=float(conf),
                                frame_id=frame_id,
                                timestamp=timestamp
                            )
                            tracked_persons.append(tracked_person)
        
        self.frame_id = frame_id
        
        return TrackingResult(
            frame_id=frame_id,
            timestamp=timestamp,
            tracked_persons=tracked_persons,
            frame=frame.copy()
        )
    
    def reset(self) -> None:
        """Reset the tracker state."""
        self.frame_id = 0
        if self.model is not None:
            # Reset tracker state by reinitializing
            self.initialize(self.config)
    
    def _convert_bbox(self, bbox: np.ndarray) -> BoundingBox:
        """
        Convert YOLO bbox format to our BoundingBox type.
        
        Args:
            bbox: YOLO bbox format (x1, y1, x2, y2)
            
        Returns:
            BoundingBox tuple (x1, y1, x2, y2)
        """
        return tuple(map(float, bbox[:4]))
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False 