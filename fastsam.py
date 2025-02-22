import supervision as sv
from ultralytics import YOLO, FastSAM
import numpy as np
import cv2
from client import Client
import yaml


def create_mask_from_image(image: cv2.typing.MatLike):
    segmentation_model = FastSAM("FastSAM-s.pt")
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    everything_results = segmentation_model(rgb_frame, device="cpu", retina_masks=True, conf=0.4, iou=0.9)
    masks = everything_results[0].masks.xy

    # Create mask overlay
    for mask in masks:
        image = cv2.polylines(image, np.int32([mask]),True,(0,255,255))
    cv2.imwrite("cutout.jpg")
    return masks


if __name__ == '__main__':
    def get_center_coordinates(xyxy):
        """Calculate center coordinates from bounding box."""
        x1, y1, x2, y2 = xyxy
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return center_x, center_y

    # Load configuration
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    VIDEO_WIDTH = config['video']['size']['width']
    VIDEO_HEIGHT = config['video']['size']['height']
    rtsp_urls = config['video']['rtsp_urls']

    # Initialize YOLOv8 segmentation model
    model = YOLO('yolov8n-seg.pt')  # Using YOLOv8 nano segmentation model
    tracker = sv.ByteTrack()

    # Add mask annotator for visualization
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    # RTSP stream configuration
    rtsp_url = "rtsp://admin:Daxhuz-kitnor-cekvi5@192.168.3.49/Preview_01_main"

    # Connect to all RTSP streams
    caps = [Client(url) for url in rtsp_urls]

    # Add MOG2 background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # Add these constants near the top
    PERSON_CLASS_ID = 0
    CONFIDENCE_THRESHOLD = 0.25
    try:
        while True:  # Changed to infinite loop for live streaming
            for cap in caps:
                ret, frame = cap.read()
                    
                # Resize frame
                print(f"Frame size: {frame.size}")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Segment image
                segmentation_model = FastSAM("FastSAM-s.pt")
                everything_results = segmentation_model(rgb_frame, device="cpu", retina_masks=True, conf=0.4, iou=0.9)
                
                if not len(everything_results):
                    continue
                masks = everything_results[0].masks.xy

            # Create mask overlay
                mask_overlay = np.zeros_like(frame, dtype=np.uint8)
                frame_with_plot = frame
                for mask in masks:
                    frame_with_plot = cv2.polylines(frame_with_plot, np.int32([mask]),True,(0,255,255))
                    # mask = (mask * 255).astype(np.uint8)
                    # mask_overlay = cv2.addWeighted(mask_overlay, 1, mask, 0.5, 0)  # Blend

                # # Overlay mask on original frameq
                # blended_frame = cv2.addWeighted(frame, 1, mask_overlay, 0.5, 0)

                # Display result
                cv2.imshow("FastSAM Segmentation", frame_with_plot)
            
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