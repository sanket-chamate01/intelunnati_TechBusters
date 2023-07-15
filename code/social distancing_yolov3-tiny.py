import time
import math
import cv2
import numpy as np
import collections

# Thresholds for confidence and distance for detecting close pairs
confid = 0.5
thresh = 0.5

# Video file
vid_file_name = 'video'
vid_path = '../data/' + vid_file_name + '.mp4'

angle_factor = 0.8
#hori_zoom_factor = 1.2

# Euclidean distance
def dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

def tan2sine(T):
    ts = abs(T/((1+T**2)**0.5))
    return ts

def tan2cosine(T):
    tc = abs(1/((1+T**2)**0.5))
    return tc

def isclose(p1,p2):
    c_dist = dist(p1[2], p2[2])
    if(p1[1]<p2[1]):
        a_w = p1[0]
        a_h = p1[1]
    else:
        a_w = p2[0]
        a_h = p2[1]
    T = 0
    try:
        T=(p2[2][1]-p1[2][1])/(p2[2][0]-p1[2][0])
    except ZeroDivisionError:
        T = 1.633123935319537e+16
    ts = tan2sine(T)
    tc = tan2cosine(T)
    dist_hori = tc * c_dist
    dist_ver = ts * c_dist
    c_calib_hor = a_w*1.3
    c_calib_ver = a_h*0.4*angle_factor
    
    if (0 < dist_hori < c_calib_hor and 0 < dist_ver < c_calib_ver):
        return 1
    else:
        return 0

# COCO dataset labels
labelsPath = "../data/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)

# YOLOv3-tiny weights and configuration files
weightsPath = "../models/yolov3-tiny.weights"
configPath = "../models/yolov3-tiny.cfg"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = list(net.getLayerNames())
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

#Initialize video capture
cap = cv2.VideoCapture(vid_path)

(W, H) = (None, None)

fl = 0
q = 0

# Queue to store processing time of frame for FPS calculation
processing_times = collections.deque()

# Main loop 
while True:
    (grabbed, frame) = cap.read()

    if not grabbed:
        break

    # Get video frame dimensions
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        FW=W
        if(W<1075):
            FW = 1075

        col = (255,255,255)
        FH = H + 210

    # Preprocess the frames
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    # Perform Object Detection
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    processing_times.append(end - start)

    # Calculate average processing time to estimate FPS
    if len(processing_times) > 200:
        processing_times.popleft()

    processing_time = np.mean(processing_times) * 1100
    fps = 1000 / processing_time
    
    boxes = []
    confidences = []
    classIDs = []
    
    # Iterate over YOLO output layer
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] == "person":
                if confidence > confid:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    #Apply Non-Maximum Suppression to remove duplicate bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

    # Process detected people bounding boxes to check for social distancing
    if len(idxs) > 0:
        status = []
        idf = idxs.flatten()
        close_pair = []
        s_close_pair = []
        center = []
        co_info = []

        # Iterate over each detected person
        for i in idf:           
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cen = [int(x + w / 2), int(y + h / 2)]
            center.append(cen)
            co_info.append([w, h, cen])
            status.append(0)

        # Find close pairs
        for i in range(len(center)):
            for j in range(len(center)):
                g = isclose(co_info[i],co_info[j])
                if g == 1:
                    close_pair.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1

        # Total number of people in risk
        total_p = len(center)
        high_risk_p = status.count(1)
        safe_p = status.count(0)
        kk = 0

        # Prints Inference time
        cv2.putText(
            img=frame,
            text=f"Inference time: {processing_time:.1f}ms",
            org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        #Prints FPS
        cv2.putText(
            img=frame,
            text=f"FPS: {fps:.1f}",
            org=(20, 70),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # Draw bounding boxes based on category of risk
        # Red for risk
        # Green for safe
        for i in idf:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if status[kk] == 1:
                cv2.putText(frame, "Violation!!!", (700, 90),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            elif status[kk] == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            kk += 1
        cv2.imshow('Social Distancing', frame)
        cv2.waitKey(1)
cap.release()
