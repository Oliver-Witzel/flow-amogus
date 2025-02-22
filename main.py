import supervision as sv
from ultralytics import YOLO
import cv2
from client import Client

VIDEO_SIZE = 600

model = YOLO('models/yolov8n-seg.pt')
tracker = sv.ByteTrack()
label_annotator = sv.LabelAnnotator()

#rtsp://admin:Daxhuz-kitnor-cekvi5@192.168.3.49/Preview_01_main
#rtsp://admin:Daxhuz-kitnor-cekvi5@192.168.3.47/Preview_01_main
#rtsp://admin:Daxhuz-kitnor-cekvi5@192.168.3.52/Preview_01_main

rtsp_url = "rtsp://admin:Daxhuz-kitnor-cekvi5@192.168.3.52/Preview_01_main"
cap = Client(rtsp_url)

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