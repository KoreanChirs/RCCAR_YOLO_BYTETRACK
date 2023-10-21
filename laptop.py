
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from typing import List
import numpy as np
from ultralytics import YOLO
import cv2
import socket
import threading
import queue
import time
import struct

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

model = YOLO("yolov8x.pt")
model.to('cuda')
CLASS_NAMES_DICT = model.model.names
CLASS_ID = [0]
byte_tracker = BYTETracker(BYTETrackerArgs())
model.predict("test.png")

# settings
LINE_START = Point(50, 1500)
LINE_END = Point(3840-50, 1500)

# Set the resolution of the video

box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
line_counter = LineCounter(start=LINE_START, end=LINE_END)
line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

# Define the host and port for the server (your laptop)
HOST = '172.20.10.3'
PORT = 5555

# Create a socket object and bind it to the host and port
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

# Accept a client connection
print('Waiting for connection...')
client_socket, address = server_socket.accept()
print(f'Connected to {address[0]}')
gogo = 0
person_Detect = False
person_id = -10
def is_go():
    global person_Detect
    global gogo
    global person_id
    if(person_Detect):
        gogo = int(input("let's get it?  1 or 0 \n"))
        if (gogo == 1):
            person_id = int(input("which person do you want to track?"))
    time.sleep(2)
    is_go()

th2 = threading.Thread(target = is_go)
th2.start()


while True:
    # Receive the length of the frame data and then the frame data itself
    data = b''
    while len(data) < 16:
        data += client_socket.recv(16 - len(data))
    frame_length = int(data)
    data = b''
    while len(data) < frame_length:
        data += client_socket.recv(frame_length - len(data))

    # Convert the received data back into an OpenCV frame
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(frame)
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )
    # filtering out detections with unwanted classes
    mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=detections),
        img_info=frame.shape,
        img_size=frame.shape
    )
    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
    detections.tracker_id = np.array(tracker_id)
    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    # format custom labels
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]
    # updating line counter
    line_counter.update(detections=detections)
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    line_annotator.annotate(frame=frame, line_counter=line_counter)
    if 0 in detections.class_id:
        print("Detect person")
        person_Detect = True
        if(gogo==1):
            if(person_id in tracker_id):
                print("go")
                for i in range(len(tracker_id)):
                    if(person_id == tracker_id[i]):
                        idx = i
                bottom_right_x = detections.xyxy[idx][0]
                bottom_right_y = detections.xyxy[idx][1]
                top_left_x = detections.xyxy[idx][2]
                top_left_y = detections.xyxy[idx][3]
                centor_x = (top_left_x + bottom_right_x)/2
                height = top_left_y-bottom_right_y
                real_centor_x = 320
                real_height = 480
                act1 = [0,0,0,0]
                if(height > 480):
                    act1[0] = 0
                else:
                    act1[0] = 20
                if (centor_x-real_centor_x < 100 and centor_x-real_centor_x > -100):
                    act1[1] = 90
                elif(centor_x > real_centor_x):
                    act1[1] = 110
                else:
                    act1[1] = 55
                data1 = ','.join(str(elem) for elem in act1)
                print(data1)
                client_socket.send(data1.encode())
            else:
                print("stop")
                act2 = [0,90,0,0]
                data2 = ','.join(str(elem) for elem in act2)
                client_socket.send(data2.encode())                     
        else:
            print("stop")
            act2 = [0,90,0,0]
            data2 = ','.join(str(elem) for elem in act2)
            client_socket.send(data2.encode())     
    else:
        act2 = [0,90,0,0]
        data2 = ','.join(str(elem) for elem in act2)
        client_socket.send(data2.encode())
        person_Detect = False
    cv2.imshow("Video",frame)
    cv2.waitKey(1)