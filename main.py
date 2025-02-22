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

# Define pose connections (COCO format keypoint pairs)
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face and neck
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12),  # Body
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (11, 12)  # Hip
]

# Initialize YOLOv8 model and separate trackers for each camera
model = YOLO(config['model']['path'])
trackers = [sv.ByteTrack() for _ in rtsp_urls]  # Create tracker for each camera
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Connect to all RTSP streams
caps = [Client(url) for url in rtsp_urls]

try:
    while True:
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            
            results = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Use camera-specific tracker
            detections = trackers[i].update_with_detections(detections)
            
            labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
            
            # First draw boxes
            frame_with_boxes = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            
            # Then add labels to the frame with boxes
            frame_with_labels = label_annotator.annotate(
                scene=frame_with_boxes,
                detections=detections,
                labels=labels
            )

            # Add pose keypoints if available in results
            if hasattr(results, 'keypoints') and results.keypoints is not None:
                frame_with_poses = frame_with_labels.copy()
                keypoints = results.keypoints.data
                
                # Add debug print
                print(f"Camera {i}, Keypoints shape: {keypoints.shape}")
                print(f"First person keypoints:\n{keypoints[0] if len(keypoints) > 0 else 'None'}")
                
                for person_keypoints in keypoints:
                    # Draw keypoint connections
                    for connection in POSE_CONNECTIONS:
                        pt1_x = int(person_keypoints[connection[0]][0])
                        pt1_y = int(person_keypoints[connection[0]][1])
                        pt2_x = int(person_keypoints[connection[1]][0])
                        pt2_y = int(person_keypoints[connection[1]][1])
                        conf1 = person_keypoints[connection[0]][2]
                        conf2 = person_keypoints[connection[1]][2]
                        
                        # More strict confidence check
                        if (conf1 > 0.5 and conf2 > 0.5 and  # Increased confidence threshold
                            0 < pt1_x < VIDEO_WIDTH and 0 < pt1_y < VIDEO_HEIGHT and
                            0 < pt2_x < VIDEO_WIDTH and 0 < pt2_y < VIDEO_HEIGHT):
                            cv2.line(frame_with_poses, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), 2)
                        
                    # Draw keypoints
                    for kp in person_keypoints:
                        x, y, conf = int(kp[0]), int(kp[1]), kp[2]
                        if conf > 0 and 0 <= x < VIDEO_WIDTH and 0 <= y < VIDEO_HEIGHT:
                            cv2.circle(frame_with_poses, (x, y), 4, (0, 0, 255), -1)
                
                frames.append(frame_with_poses)
            else:
                frames.append(frame_with_labels)
        
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