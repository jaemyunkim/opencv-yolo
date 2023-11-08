import os
import requests
from pathlib import Path
import numpy as np
import cv2

from utils import get_yolo_weights


def main():
    YOLO_MODEL = "yolov3"
    YOLO_DATA = "coco"

    # get yolo model files
    get_yolo_weights(YOLO_MODEL)

    # initialize YOLO model
    net, classes, output_layers = initialize(YOLO_MODEL, YOLO_DATA)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # read image
    image = cv2.imread("samples/image1.jpg")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape

    # prepare run the detection
    blob = cv2.dnn.blobFromImage(img, 1/256, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # run the forward pass to get a predict result
    outs = net.forward(output_layers)
    
    # post-processing
    indexes, class_ids, confidences, boxes = postprocessing(outs, (width, height))

    # draw result
    resImg = image.copy()
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            print(x, y, w, h)
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(resImg, (x, y), ((x + w), (y + h)), color, 2)
            cv2.putText(resImg, label + " " + confidence, (x, y + 20), font, 2, (0, 255, 0), 2)

    # write the result to a file
    cv2.imwrite("result.png", resImg)
    cv2.waitKey(1)


def initialize(yolo_model, yolo_data):
    yolo_model = yolo_model.lower()
    yolo_data = yolo_data.lower()

    net = cv2.dnn.readNet(f"models/{yolo_model}.weights", f"models/{yolo_model}.cfg")
    classes = []
    with open(f"models/{yolo_data}.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers


def postprocessing(outs, size):
    width = size[0]
    height = size[1]

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Object detection
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # coordinate
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    return indexes, class_ids, confidences, boxes


if __name__ == "__main__":
    main()
    