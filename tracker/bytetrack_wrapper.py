"""
BoT-SORT tracker wrapper using Ultralytics YOLO for person detection and tracking.
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

class BoTSORTWrapper(BaseTracker):
    """Wrapper for BoT-SORT tracking algorithm using Ultralytics YOLO."""
    
    def __init__(self, config: dict):
        """
        Initialize BoT-SORT wrapper.
        
        Args:
            config: Dictionary containing BoT-SORT configuration parameters
        """
        self.config = config
        self.model = None
        self.frame_id = 0
        self.initialize(config)
    
    def initialize(self, config: dict) -> None:
        """
        Initialize BoT-SORT with YOLO model.
        
        Args:
            config: Dictionary containing parameters:
                - model_path: Path to YOLO model (default: 'yolov8n.pt')
                - conf_threshold: Detection confidence threshold (default: 0.3)
                - iou_threshold: NMS IOU threshold (default: 0.5)
                - device: Device to run on ('cpu', 'cuda', etc.)
                - tracker: Tracker type ('botsort.yaml')
        """
        model_path = config.get('model_path', 'yolov8n.pt')
        self.conf_threshold = config.get('conf_threshold', 0.3)
        self.iou_threshold = config.get('iou_threshold', 0.5)
        self.device = config.get('device', 'cuda' if self._cuda_available() else 'cpu')
        tracker_config = config.get('tracker', 'botsort.yaml')
        
        # Initialize YOLO model with BoT-SORT tracker
        self.model = YOLO(model_path)
        
        # Configure tracker
        self.model.track(
            source=np.zeros((640, 640, 3), dtype=np.uint8),  # Dummy frame for initialization
            tracker=tracker_config,
            persist=True,
            verbose=False
        )
    
    def update(self, frame: np.ndarray, frame_id: FrameID) -> TrackingResult:
        """
        Update tracking with a new frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            frame_id: Frame identifier
            
        Returns:
            TrackingResult containing tracked persons and frame
        """
        timestamp = datetime.now()
        tracked_persons = []
        
        if self.model is not None:
            # Run YOLO detection and tracking
            results = self.model.track(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                classes=[0],  # Only detect persons (class 0 in COCO)
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
                        tracked_person = TrackedPerson(
                            track_id=track_id,
                            bbox=self._convert_bbox(bbox),
                            confidence=float(conf),
                            timestamp=timestamp,
                            features={}  # Will be populated by feature extractors
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

# Keep ByteTrackWrapper as an alias for backward compatibility
ByteTrackWrapper = BoTSORTWrapper 