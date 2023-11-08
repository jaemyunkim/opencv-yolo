import os
import requests
from pathlib import Path
import numpy as np
import cv2

from utils import get_yolo_weights


def main():
    YOLO_MODEL = "yolov3"
    YOLO_DATA = "coco"

    # initialize YOLO model
    net, classes, output_layers = initialize(YOLO_MODEL, YOLO_DATA)

    # read image
    image = cv2.imread("samples/image1.jpg")
    height, width, channels = image.shape

    # predict objects
    indexes, class_ids, confidences, boxes = predict(image, net, output_layers, (width, height))

    # draw result
    resImg = image.copy()
    drawPred(resImg, indexes, class_ids, confidences, boxes, classes)

    # write and display the result to a file
    cv2.imwrite("result.png", resImg)
    cv2.imshow("image", image)
    cv2.imshow("result", resImg)
    cv2.waitKey(0)


def predict(image, net, output_layers, size):
    width = size[0]
    height = size[1]

    # revert channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # prepare run the detection
    blob = cv2.dnn.blobFromImage(image, 1/256, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # run the forward pass to get a predict result
    outs = net.forward(output_layers)
    
    # post-processing
    indexes, class_ids, confidences, boxes = postprocessing(outs, (width, height))

    return indexes, class_ids, confidences, boxes


def initialize(model_name, data_name):
    model_name = model_name.lower()
    data_name = data_name.lower()

    # get yolo model files
    get_yolo_weights(model_name)

    # load yolo model
    net = cv2.dnn.readNet(f"models/{model_name}.weights", f"models/{model_name}.cfg")
    classes = []
    with open(f"models/{data_name}.names", "r") as f:
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


def drawPred(image, indexes, class_ids, confidences, boxes, classes):
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            print(x, y, w, h)
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[class_ids[i]]
            # color = colors[i]
            cv2.rectangle(image, (x, y), ((x + w), (y + h)), color, 2)
            cv2.putText(image, label + " " + confidence, (x, y + 20), font, 2, (0, 255, 0), 2)


if __name__ == "__main__":
    main()
    