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
        self._model_name = model_name.lower()
        self._data_name = data_name.lower()

        self._classes = []
        self._colors = []
        self.reset_class_filter()

        # get yolo model files
        self._input_width, self._input_height = get_yolo_weights(self._model_name)

        # initialize yolo
        self._output_layers = ()
        self._initialize()

        self._outs = ()
        self._class_ids = []
        self._confidences = []
        self._boxes = []
        self._indexes = []

        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._colors = np.random.uniform(0, 255, size=(len(self._classes), 3))


    def classes(self):
        return self._classes
    

    def class_filter(self, classes = []):
        pass


    def reset_class_filter(self):
        self._class_filter = []
    

    def predict(self, image):
        self._height, self._width = image.shape[:2]

        # revert channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # prepare run the detection
        # blob = cv2.dnn.blobFromImage(image, 1/256, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        blob = cv2.dnn.blobFromImage(image, 1/256, (640, 640), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)

        # run the forward pass to get a predict result
        self._outs = self.net.forward(self._output_layers)
        
        # post-processing
        self.postprocessing()


    def _initialize(self):
        self._classes = []
        with open(f"models/{self._data_name}.names", "r") as f:
            self._classes = [line.strip() for line in f.readlines()]

        # load yolo model
        if self._model_name.startswith("yolov5"):
            self.net = cv2.dnn.readNet(f"models/{self._model_name}.onnx")
        else:
            self.net = cv2.dnn.readNet(f"models/{self._model_name}.weights", f"models/{self._model_name}.cfg")
        # layer_names = self.net.getLayerNames()
        # self._output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self._output_layers = self.net.getUnconnectedOutLayersNames()


    def postprocessing(self):
        self._boxes = []
        self._confidences = []
        self._class_ids = []
    
        if self._model_name.startswith("yolov5"):
            self._postprocessing_recent_yolo()
        else:
            self._postprocessing_original_yolo()

        self._indexes = cv2.dnn.NMSBoxes(self._boxes, self._confidences, 0.5, 0.4)


    def _postprocessing_original_yolo(self):
        for out in self._outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    # Object detection
                    center_x = int(detection[0] * self._width)
                    center_y = int(detection[1] * self._height)
                    w = int(detection[2] * self._width)
                    h = int(detection[3] * self._height)

                    # coordinate
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    self._boxes.append([x, y, w, h])
                    self._confidences.append(float(confidence))
                    self._class_ids.append(class_id)

    
    def _postprocessing_recent_yolo(self):
        outs = self._outs[0]
        x_factor = self._width / self._input_width
        y_factor = self._height / self._input_height

        for detection in self._outs[0][0]:
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
                    self._boxes.append([x, y, w, h])
                    self._confidences.append(float(confidence))
                    self._class_ids.append(class_id)


    def drawPred(self, image):
        if len(self._indexes) > 0:
            for i in self._indexes.flatten():
                x, y, w, h = self._boxes[i]
                # print(x, y, w, h)
                label = str(self._classes[self._class_ids[i]])
                confidence = str(round(self._confidences[i], 2))
                color = self._colors[self._class_ids[i]]
                # color = colors[i]
                cv2.rectangle(image, (x, y), ((x + w), (y + h)), color, 2)

                textLoc = (x, y + 15)
                text = label + " " + confidence
                fontScale = 0.5
                thickness = 1
                size, baseline = cv2.getTextSize(text, self._font, fontScale, thickness)
                cv2.rectangle(image, (x, y), (textLoc[0] + size[0], y + baseline + size[1]), color, -1)
                textColor = [0, 0, 0] if np.average(color) > 127 else [255, 255, 255]
                cv2.putText(image, text, textLoc, self._font, fontScale, textColor, thickness)


if __name__ == "__main__":
    main()
    