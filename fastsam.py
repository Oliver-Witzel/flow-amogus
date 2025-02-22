import supervision as sv
from ultralytics import YOLO, FastSAM
import numpy as np
import cv2
from client_cpu import Client
from ultralytics.utils.plotting import Annotator

REAL_VIDEO_SIZE = 3008
VIDEO_SIZE = 600
def get_coordinate_in_frame_size(coordinate):
    return int((coordinate/REAL_VIDEO_SIZE)*VIDEO_SIZE)

def get_center_coordinates(xyxy):
    """Calculate center coordinates from bounding box."""
    x1, y1, x2, y2 = xyxy
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

# Initialize YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')  # Using YOLOv8 nano segmentation model
tracker = sv.ByteTrack()

# Add mask annotator for visualization
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

# RTSP stream configuration
rtsp_url = "rtsp://admin:Daxhuz-kitnor-cekvi5@192.168.3.49/Preview_01_main"
cap = Client(rtsp_url)

X1 = get_coordinate_in_frame_size(2100)
X2 = get_coordinate_in_frame_size(2500)
Y1 = get_coordinate_in_frame_size(1500)
Y2 = get_coordinate_in_frame_size(2250)

# Add MOG2 background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Add these constants near the top
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.25

try:
    while True:  # Changed to infinite loop for live streaming
        ret, frame = cap.read()
            
        # Resize frame
        print(f"Frame size: {frame.size}")
        break
        frame = cv2.resize(frame, (VIDEO_SIZE, VIDEO_SIZE))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Segment image
        segmentation_model = FastSAM("FastSAM-s.pt")
        results = model(rgb_frame, device="cpu", retina_masks=True, imgsz=VIDEO_SIZE, conf=0.4, iou=0.9)
        masks = results[0].masks  # Extract masks from results

        # Draw masks on the frame
        print(results)
        # annotator = Annotator(frame)
        # if masks is not None:
        #     for mask in masks:
        #         annotator.draw_mask(mask)
        cv2.imshow("Frame", frame)
        
        # # Apply background subtraction first
        # fgMask = backSub.apply(frame)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        # fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        
        # # Get moving objects
        # moving_objects = cv2.bitwise_and(frame, frame, mask=fgMask)
        
        # # Detect all objects in the frame with segmentation
        # results = model(frame)[0]
        # detections = sv.Detections.from_ultralytics(results)
        
        # # Create a mask for humans using segmentation masks
        # human_mask = np.zeros_like(fgMask)
        # person_detections = detections[detections.class_id == PERSON_CLASS_ID]
        
        # if len(person_detections) > 0 and person_detections.mask is not None:  # First check if we have any detections
        #     # Combine all person masks
        #     for mask in person_detections.mask:
        #         # Resize mask to match frame size
        #         mask = cv2.resize(mask.astype(np.uint8), (VIDEO_SIZE, VIDEO_SIZE))
        #         human_mask = cv2.bitwise_or(human_mask, mask * 255)
        
        # # Dilate human mask slightly to ensure complete coverage
        # human_mask = cv2.dilate(human_mask, kernel, iterations=2)
        
        # # Invert human mask to get non-human regions
        # non_human_mask = cv2.bitwise_not(human_mask)
        
        # # Combine with motion mask to get moving non-human objects
        # final_mask = cv2.bitwise_and(fgMask, non_human_mask)
        # other_moving_objects = cv2.bitwise_and(frame, frame, mask=final_mask)
        
        # # Update tracking
        # detections = tracker.update_with_detections(person_detections)
        # for i, tracker_id in enumerate(detections.tracker_id):
        #     labels.append(tracker_id)
        #     if tracker_id not in has_visit_cashier:
        #         has_visit_cashier[tracker_id] = False
        #     if has_visit_cashier[tracker_id]:
        #         continue
            
        #     #x1, y1, x2, y2 = detections.xyxy[i]
        #     center_x, center_y = get_center_coordinates(detections.xyxy[i])
        #     #if ((x1 > X1 and x1 < X2) and ((y1 > Y1 and y1 < Y2))) or ((x2 > X1 and x2 < X2) and (y2 > Y1 and y2 < Y2)):
        #     if (center_x < X2 and center_x > X1) and (center_y < Y2 and center_y > Y1):
        #         has_visit_cashier[tracker_id] = True
        
        # # Create labels with tracker IDs
        # labels = [f"#{tracker_id}-{has_visit_cashier.get(tracker_id) or 'noinfo'}" for tracker_id in detections.tracker_id]
        
        # # Annotate frame with segmentation masks
        # annotated_frame = frame.copy()
        # if detections.mask is not None and len(detections.mask) > 0:  # Add null check
        #     annotated_frame = mask_annotator.annotate(
        #         scene=annotated_frame, 
        #         detections=detections
        #     )
        # annotated_frame = label_annotator.annotate(
        #     scene=annotated_frame, 
        #     detections=detections, 
        #     labels=labels
        # )
        
        # # Show both the tracking and motion detection
        # cv2.imshow('Live Tracking', annotated_frame)
        # cv2.imshow('Motion Detection', moving_objects)
        # cv2.imshow('Other Moving Objects', other_moving_objects)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    # Update cleanup section
    cap.release()
    cv2.destroyAllWindows()