import numpy as np
import cv2

from .detect_config import MIN_CONF, NMS_THRESH


def detect_things(frame, net, ln, thing_indexes=None):
    # grab the dimensions of the frame and  initialize the list of
    # results
    if thing_indexes is None:
        thing_indexes = []
    (H, W) = frame.shape[:2]
    results = []

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, centroids, and confidences, respectively
    boxes = []
    centroids = []
    confidences = []
    label_indexes = []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # filter detections by (1) ensuring that the object
            # detected was a thing and (2) that the minimum
            # confidence is met
            if class_id in thing_indexes and confidence > MIN_CONF:
                # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
                label_indexes.append(class_id)

    # apply non-maximum suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update our results list to consist of the thing prediction probability, bounding box coordinates,
            # and the centroid
            detected_object = (confidences[i], (x, y, x + w, y + h), centroids[i], label_indexes[i])
            results.append(detected_object)

    # return the list of results
    return results