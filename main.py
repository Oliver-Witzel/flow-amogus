import supervision as sv
from ultralytics import YOLO
import cv2
from client import Client
import yaml

# Load configuration
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

VIDEO_WIDTH = config['video']['size']['width']
VIDEO_HEIGHT = config['video']['size']['height']
rtsp_urls = config['video']['rtsp_urls']

# Initialize YOLOv8 model and tracker
model = YOLO(config['model']['path'])
tracker = sv.ByteTrack()
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
            
            annotated_frame = label_annotator.annotate(
                scene=frame.copy(), 
                detections=detections, 
                labels=labels
            )
            
            frames.append(annotated_frame)
        
        # Combine frames horizontally
        combined_frame = cv2.hconcat(frames)
        
        # Display combined frame
        cv2.imshow('All Cameras', combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()