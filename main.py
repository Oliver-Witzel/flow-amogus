import supervision as sv
from ultralytics import YOLO
import cv2
from client import Client
import yaml

# Load configuration
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

VIDEO_SIZE = config['video']['size']
rtsp_urls = config['video']['rtsp_urls']

# Initialize YOLOv8 model and tracker
model = YOLO(config['model']['path'])
tracker = sv.ByteTrack()
label_annotator = sv.LabelAnnotator()

# Connect to RTSP stream
cap = Client(rtsp_urls[0])

try:
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (VIDEO_SIZE, VIDEO_SIZE))
        
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        detections = tracker.update_with_detections(detections)
        
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        
        annotated_frame = label_annotator.annotate(
            scene=frame.copy(), 
            detections=detections, 
            labels=labels
        )
        
        cv2.imshow('Live Tracking', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()