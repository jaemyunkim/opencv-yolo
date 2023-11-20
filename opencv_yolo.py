import os
import requests
from pathlib import Path
import numpy as np
import cv2

from utils import get_yolo_weights


def main():
    YOLO_MODEL = "yolov5s"
    YOLO_DATA = "coco"

    # initialize YOLO model
    od = opencv_yolo(YOLO_MODEL, YOLO_DATA)

    # read image
    image = cv2.imread("samples/image1.jpg")

    # predict objects
    od.predict(image)

    # draw result
    resImg = image.copy()
    od.drawPred(resImg)

    # write and display the result to a file
    cv2.imwrite("result.png", resImg)
    cv2.imshow("image", image)
    cv2.imshow("result", resImg)
    cv2.waitKey(0)


class opencv_yolo:
    def __init__(self, model_name, data_name):
        self.model_name = model_name.lower()
        self.data_name = data_name.lower()

        self.colors = []

        # get yolo model files
        self.input_width, self.input_height = get_yolo_weights(self.model_name)

        # initialize yolo
        self._initialize(self.model_name, self.data_name)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))


    def predict(self, image):
        self.image = image.copy()
        self.height, self.width = image.shape[:2]

        # revert channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # prepare run the detection
        # blob = cv2.dnn.blobFromImage(image, 1/256, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        blob = cv2.dnn.blobFromImage(image, 1/256, (640, 640), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)

        # run the forward pass to get a predict result
        self.outs = self.net.forward(self.output_layers)
        
        # post-processing
        self.postprocessing()


    def _initialize(self, model_name, data_name):
        self.classes = []
        with open(f"models/{data_name}.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # load yolo model
        if model_name.startswith("yolov5"):
            self.net = cv2.dnn.readNet(f"models/{model_name}.onnx")
        else:
            self.net = cv2.dnn.readNet(f"models/{model_name}.weights", f"models/{model_name}.cfg")
        layer_names = self.net.getLayerNames()
        # self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.output_layers = self.net.getUnconnectedOutLayersNames()


    def postprocessing(self):
        self.class_ids = []
        self.confidences = []
        self.boxes = []

        if self.model_name.startswith("yolov5"):
            self._postprocessing_recent_yolo()
        else:
            self._postprocessing_original_yolo()

        self.indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)


    def _postprocessing_original_yolo(self):
        for out in self.outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    # Object detection
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)

                    # coordinate
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    self.boxes.append([x, y, w, h])
                    self.confidences.append(float(confidence))
                    self.class_ids.append(class_id)

    
    def _postprocessing_recent_yolo(self):
        outs = self.outs[0]
        x_factor = self.width / self.input_width
        y_factor = self.height / self.input_height

        for detection in self.outs[0][0]:
            confidence = detection[4]

            # Discard bad detections and continue.
            if confidence >= 0.5:   # CONFIDENCE_THRESHOLD
                scores = detection[5:]
                class_id = np.argmax(scores)
                class_score = scores[class_id]
                
                if class_score > 0.5:   # SCORE_THRESHOLD
                    cx, cy, w, h = detection[:4]
                    x = int((cx - w / 2) * x_factor)    # left
                    y = int((cy - h / 2) * y_factor)    # right
                    w = int(w * x_factor)
                    h = int(h * y_factor)
                    self.boxes.append([x, y, w, h])
                    self.confidences.append(float(confidence))
                    self.class_ids.append(class_id)


    def drawPred(self, image):
        if len(self.indexes) > 0:
            for i in self.indexes.flatten():
                x, y, w, h = self.boxes[i]
                print(x, y, w, h)
                label = str(self.classes[self.class_ids[i]])
                confidence = str(round(self.confidences[i], 2))
                color = self.colors[self.class_ids[i]]
                # color = colors[i]
                cv2.rectangle(image, (x, y), ((x + w), (y + h)), color, 2)
                cv2.putText(image, label + " " + confidence, (x, y + 15), self.font, 0.5, (0, 255, 0), 1)


if __name__ == "__main__":
    main()
    