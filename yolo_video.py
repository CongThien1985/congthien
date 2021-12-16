# USAGE:
# python yolo_video.py --input traffic_monitoring.mp4 --output traffic_monitoring_output.avi --yolo yolo-coco
# python yolo_video.py --input video_street_view.mp4 --output video_street_view_output.avi --yolo yolo-coco

import os
# os.chdir('D:\\YOLO\\Yolo-Vehicle-Counter')
# import the necessary packages
from vehicle import detect_config as config
from vehicle.detection import detect_things
# from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import math
import time
import dlib

WIDTH = 1280
HEIGHT = 720
rectangleColor = (0, 255, 0)
frameCounter = 0
currentCarID = 0
fps = 0

carTracker = {}
carNumbers = {}
carLocation1 = {}
carLocation2 = {}
speed = [None] * 1000


# Function calculate speed vehicle
def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    # pixel per meter
    ppm = 8.8
    d_meters = d_pixels / ppm
    #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    fps = 20
    speed = d_meters * fps * 3.6
    return speed


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
# ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if config.USE_GPU:
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()  # list LayerNames
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
list_of_vehicles = ["car", "motorbike"]
list_of_vehicle_ids = list(map(lambda x: LABELS.index(x), list_of_vehicles))

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# loop over the frames from the video stream
while True:
    start_time = time.time()
    # read the next frame from the file
    rc, image = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if type(image) == type(None):
        break

    # resize the frame and then detect vehicle
    image = cv2.resize(image, (WIDTH, HEIGHT))
    resultImage = image.copy()
    frameCounter = frameCounter + 1

    # add to
    carIDtoDelete = []
    for carID in carTracker.keys():
        trackingQuality = carTracker[carID].update(image)

        if trackingQuality < 7:
            carIDtoDelete.append(carID)

    for carID in carIDtoDelete:
        print('Removing carID ' + str(carID) + ' from list of trackers.')
        print('Removing carID ' + str(carID) + ' previous location.')
        print('Removing carID ' + str(carID) + ' current location.')
        carTracker.pop(carID, None)
        carLocation1.pop(carID, None)
        carLocation2.pop(carID, None)

    # loop over the results
    if not (frameCounter % 30):
        results = detect_things(image, net, ln, thing_indexes=list_of_vehicle_ids)
        for (i, (prob, bbox, centroid, label_index)) in enumerate(results):
            # extract the bounding box and centrqoid coordinates, then initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            x = int(startX)
            y = int(startY)
            w = int(endX-startX)
            h = int(endY-startY)
            (cX, cY) = centroid
            x_bar = x + 0.5 * w
            y_bar = y + 0.5 * h

            matchCarID = None
            for carID in carTracker.keys():
                trackedPosition = carTracker[carID].get_position()

                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())

                t_x_bar = t_x + 0.5 * t_w
                t_y_bar = t_y + 0.5 * t_h

                if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (
                        y <= t_y_bar <= (y + h))):
                    matchCarID = carID

            if matchCarID is None:
                print('Creating new tracker ' + str(currentCarID))
                # Create the correlation tracker - the object needs to be initialized
                # before it can be used
                tracker = dlib.correlation_tracker()
                # Start a track on the juice box. If you look at the first frame you
                # will see that the juice box is contained within the bounding
                # box (x, y, x+w, y+h).
                tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                carTracker[currentCarID] = tracker
                carLocation1[currentCarID] = [x, y, w, h]

                currentCarID = currentCarID + 1

    for carID in carTracker.keys():
        trackedPosition = carTracker[carID].get_position()

        t_x = int(trackedPosition.left())
        t_y = int(trackedPosition.top())
        t_w = int(trackedPosition.width())
        t_h = int(trackedPosition.height())

        cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

        # speed estimation
        carLocation2[carID] = [t_x, t_y, t_w, t_h]

    end_time = time.time()

    if not (end_time == start_time):
        fps = 1.0 / (end_time - start_time)

    for i in carLocation1.keys():
        if frameCounter % 1 == 0:
            [x1, y1, w1, h1] = carLocation1[i]
            [x2, y2, w2, h2] = carLocation2[i]

            # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
            carLocation1[i] = [x2, y2, w2, h2]

            # print 'new previous location: ' + str(carLocation1[i])
            if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                if (speed[i] == None or speed[i] == 0):
                    speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                # if y1 > 275 and y1 < 285:
                if speed[i] != None and y1 >= 0:
                    cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1 / 2), int(y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # check to see if the output frame should be displayed to our screen
    cv2.imshow('result', resultImage)

    if cv2.waitKey(33) == 27:
        break
    # if an output video file path has been supplied and the video writer has not been initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (resultImage.shape[1], resultImage.shape[0]), True)

    # if the video writer is not None, write the frame to the output video file
    if writer is not None:
        writer.write(resultImage)
cv2.destroyAllWindows()
