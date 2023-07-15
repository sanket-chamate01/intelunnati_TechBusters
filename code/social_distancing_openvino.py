import collections
from pathlib import Path
import sys
import time
import numpy as np
import cv2
from IPython import display
import matplotlib.pyplot as plt
from openvino.runtime import Core
from utils import notebook_utils as utils
from deepsort_utils.tracker import Tracker
from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils.detection import Detection, compute_color_for_labels, xywh_to_xyxy, xywh_to_tlwh, tlwh_to_xyxy
import math
from itertools import combinations

def is_close(p1, p2):
    # Calculates Euclidean Distance between two 2d points
    dst = math.sqrt(p1**2 + p2**2)
    return dst 

def convertBack(x, y, w, h):
    # Converts center coordinates to rectangle coordinates
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

# Load Model
ie_core = Core()
class Model:
    def __init__(self, model_path, batchsize=1, device="AUTO"):
        self.model = ie_core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]

        for layer in self.model.inputs:
            input_shape = layer.partial_shape
            input_shape[0] = batchsize
            self.model.reshape({layer: input_shape})
        self.compiled_model = ie_core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, input):
        result = self.compiled_model(input)[self.output_layer]
        return result

detection_model_path = '../models/person-detection-0202.xml'
reidentification_model_path = '../models/person-reidentification-retail-0287.xml'
detector = Model(detection_model_path)
extractor = Model(reidentification_model_path, -1)

# Single image preprocessing
def preprocess(frame, height, width):
    resized_image = cv2.resize(frame, (width, height))
    resized_image = resized_image.transpose((2, 0, 1))
    input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
    return input_image


def batch_preprocess(img_crops, height, width):
    img_batch = np.concatenate([
        preprocess(img, height, width)
        for img in img_crops
    ], axis=0)
    return img_batch


def process_results(h, w, results, thresh=0.5):
    detections = results.reshape(-1, 7)
    boxes = []
    labels = []
    scores = []
    for i, detection in enumerate(detections):
        _, label, score, xmin, ymin, xmax, ymax = detection
        # Filter detected objects.
        if score > thresh:
            # Create a box with pixels coordinates from the box with normalized coordinates [0,1].
            boxes.append(
                [(xmin + xmax) / 2 * w, (ymin + ymax) / 2 * h, (xmax - xmin) * w, (ymax - ymin) * h]
            )
            labels.append(int(label))
            scores.append(float(score))

    if len(boxes) == 0:
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        labels = np.array([])
    return np.array(boxes), np.array(scores), np.array(labels)

# Function to draw bounding boxes in original image
def draw_boxes(img, bbox, identities=None):
    centroid_dict = dict() 						
    objectId = 0	
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = (0,255,0) # Green Color
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        centroid_dict[objectId] = (int((x1+x2)/2), int((y1+y2)/2), x1, y1, x2, y2) # Create dictionary of tuple with 'objectId' as the index center points and bbox
        objectId += 1 #Increment the index for each detection

        # prints a rectangle with label on screen
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 + t_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            1.6,
            [255, 255, 255],
            2
        )
        
    red_zone_list = [] # consist the detected people's id number that violate social distancing 
    red_line_list = [] # consist the detected people's centroid value that violate social distancing 
    for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]   # Check the difference between centroid x: 0, y :1
            distance = is_close(dx, dy) 			
            if distance < 75.0:						
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)   # Add Id to a list
                    red_line_list.append(p1[0:2])   #  Add points to the list
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)	
                    red_line_list.append(p2[0:2])
                     
    for idx, box in centroid_dict.items():  
          if idx in red_zone_list:  
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2) # Create Red bounding boxes 
          else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # Create Green bounding boxes

    text = "Violations: %s" % str(len(red_zone_list))   # Count number of Violations
    location = (10,430)												
    cv2.putText(img, text, location, cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

    # Declare region in the frame to be High Risk Zone if people violating social distance
    # is more than or equal to 60% of total people in the frame at that time and Moderate Risk
    # Zone if the number is more than or equal to 40%
    if len(red_zone_list) >= 0.6*len(centroid_dict):
        cv2.putText(img, "High Risk Zone!!!", (330,430), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    elif len(red_zone_list) >= 0.4*len(centroid_dict):
        cv2.putText(img, "Moderate Risk Zone!!!", (330,430), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)  
    return img

# Calculates cosine distance of 2 vectors
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

# Main Processing function 
def run_person_tracking(source=0, flip=False, use_popup=True, skip_first_frames=0):
    output_size = (700, 450)  # Output frame size
    output_fps = 24  # Output FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for video saving
    out = cv2.VideoWriter(output_file, fourcc, output_fps, output_size)
    
    player = None
    try:
        # Create a video player to play with target fps. It helps in smooth playback and testing
        player = utils.VideoPlayer(
            source=source, size=(700, 450), flip=flip, fps=24, skip_first_frames=skip_first_frames
        )
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(
                winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
            )

        processing_times = collections.deque()
        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            # Resize the image and change dims to fit neural network input.
            h, w = frame.shape[:2]
            input_image = preprocess(frame, detector.height, detector.width)

            # Measure processing time.
            start_time = time.time()
            # Get the results.
            output = detector.predict(input_image)
            stop_time = time.time()
            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1100
            fps = 1000 / processing_time

            # Get poses from detection results.
            bbox_xywh, score, label = process_results(h, w, results=output)
            
            img_crops = []
            for box in bbox_xywh:
                x1, y1, x2, y2 = xywh_to_xyxy(box, h, w)
                img = frame[y1:y2, x1:x2]
                img_crops.append(img)

            # Get reidentification feature of each person.
            if img_crops:
                # preprocess
                img_batch = batch_preprocess(img_crops, extractor.height, extractor.width)
                features = extractor.predict(img_batch)
            else:
                features = np.array([])

            # Wrap the detection and reidentification results together
            bbox_tlwh = xywh_to_tlwh(bbox_xywh)
            detections = [
                Detection(bbox_tlwh[i], features[i])
                for i in range(features.shape[0])
            ]

            # predict the position of tracking target 
            tracker.predict()

            # update tracker
            tracker.update(detections)

            # update bbox identities
            outputs = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1, y1, x2, y2 = tlwh_to_xyxy(box, h, w)
                track_id = track.track_id
                outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int32))
            if len(outputs) > 0:
                outputs = np.stack(outputs, axis=0)

            # draw box for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                frame = draw_boxes(frame, bbox_xyxy, identities)

            # Show Inference time on screen
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(255, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            # Show FPS(Frames Per Second) on screen
            cv2.putText(
                img=frame,
                text=f"FPS: {fps:.1f}",
                org=(20, 70),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(255, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            
            if use_popup:
                cv2.imshow(winname=title, mat=frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27 or key == 'q':
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(
                    ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                )
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)

            out.write(frame) # writes frames to video file 

    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()

video_file_name = 'street'
output_file = '../demo_videos/' + video_file_name + '_output.mp4'

NN_BUDGET = 100
MAX_COSINE_DISTANCE = 0.6  # threshold of matching object
metric = NearestNeighborDistanceMetric(
    "cosine", MAX_COSINE_DISTANCE, NN_BUDGET
)
tracker = Tracker(
    metric,
    max_iou_distance=0.7,
    max_age=70,
    n_init=3
)

video_file = '../data/' + video_file_name + '.mp4'
#run_person_tracking(source=0, flip=True, use_popup=False)
run_person_tracking(source=video_file, flip=False, use_popup=False)
