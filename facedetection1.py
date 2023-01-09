import copy
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import CvFpsCalc
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--model_selection", type=int, default=0)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument('--use_brect', action='store_true')
    args = parser.parse_args()
    return args
def main():
    
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    model_selection = args.model_selection
    min_detection_confidence = args.min_detection_confidence
    use_brect = args.use_brect
    
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_detection_confidence,
    )
    
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    while True:
        display_fps = cvFpsCalc.get()
        
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  
        debug_image = copy.deepcopy(image)
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_detection.process(image)
        
        if results.detections is not None:
            for detection in results.detections:
                
                debug_image = draw_detection(debug_image, detection)
        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        
        key = cv.waitKey(1)
        if key == 27:  
            break
        
        cv.imshow('MediaPipe Face Detection Demo', debug_image)
    cap.release()
    cv.destroyAllWindows()
def draw_detection(image, detection):
    image_width, image_height = image.shape[1], image.shape[0]
    
    
    
    
    
    
    
    
    bbox = detection.location_data.relative_bounding_box
    bbox.xmin = int(bbox.xmin * image_width)
    bbox.ymin = int(bbox.ymin * image_height)
    bbox.width = int(bbox.width * image_width)
    bbox.height = int(bbox.height * image_height)
    cv.rectangle(image, (int(bbox.xmin), int(bbox.ymin)),
                 (int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height)),
                 (0, 255, 0), 2)
    
    cv.putText(
        image,
        str(detection.label_id[0]) + ":" + str(round(detection.score[0], 3)),
        (int(bbox.xmin), int(bbox.ymin) - 20), cv.FONT_HERSHEY_SIMPLEX, 1.0,
        (0, 255, 0), 2, cv.LINE_AA)
    
    keypoint0 = detection.location_data.relative_keypoints[0]
    keypoint0.x = int(keypoint0.x * image_width)
    keypoint0.y = int(keypoint0.y * image_height)
    cv.circle(image, (int(keypoint0.x), int(keypoint0.y)), 5, (0, 255, 0), 2)
    
    keypoint1 = detection.location_data.relative_keypoints[1]
    keypoint1.x = int(keypoint1.x * image_width)
    keypoint1.y = int(keypoint1.y * image_height)
    cv.circle(image, (int(keypoint1.x), int(keypoint1.y)), 5, (0, 255, 0), 2)
    
    keypoint2 = detection.location_data.relative_keypoints[2]
    keypoint2.x = int(keypoint2.x * image_width)
    keypoint2.y = int(keypoint2.y * image_height)
    cv.circle(image, (int(keypoint2.x), int(keypoint2.y)), 5, (0, 255, 0), 2)
    
    keypoint3 = detection.location_data.relative_keypoints[3]
    keypoint3.x = int(keypoint3.x * image_width)
    keypoint3.y = int(keypoint3.y * image_height)
    cv.circle(image, (int(keypoint3.x), int(keypoint3.y)), 5, (0, 255, 0), 2)
    
    keypoint4 = detection.location_data.relative_keypoints[4]
    keypoint4.x = int(keypoint4.x * image_width)
    keypoint4.y = int(keypoint4.y * image_height)
    cv.circle(image, (int(keypoint4.x), int(keypoint4.y)), 5, (0, 255, 0), 2)
    
    keypoint5 = detection.location_data.relative_keypoints[5]
    keypoint5.x = int(keypoint5.x * image_width)
    keypoint5.y = int(keypoint5.y * image_height)
    cv.circle(image, (int(keypoint5.x), int(keypoint5.y)), 5, (0, 255, 0), 2)
    return image
if __name__ == '__main__':
    main()
