import supervision as sv
from ultralytics import YOLO
import cv2
from client import Client
import yaml
import numpy as np

# Load configuration
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

VIDEO_WIDTH = config['video']['size']['width']
VIDEO_HEIGHT = config['video']['size']['height']
rtsp_urls = config['video']['rtsp_urls']

# Initialize YOLOv8 model and tracker
model = YOLO(config['model']['path'])
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Connect to all RTSP streams
caps = [Client(url) for url in rtsp_urls]

try:
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            detections = tracker.update_with_detections(detections)
            
            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
            
            # First draw boxes
            frame_with_boxes = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            
            # Then add labels to the frame with boxes
            annotated_frame = label_annotator.annotate(
                scene=frame_with_boxes,
                detections=detections,
                labels=labels
            )
            
            frames.append(annotated_frame)
        
        # Create black frame for the 4th position if needed
        if len(frames) < 4:
            black_frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
            frames.append(black_frame)
        
        # Create 2x2 grid
        top_row = cv2.hconcat([frames[0], frames[1]])
        bottom_row = cv2.hconcat([frames[2], frames[3]])
        combined_frame = cv2.vconcat([top_row, bottom_row])
        
        # Display combined frame
        cv2.imshow('All Cameras', combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()