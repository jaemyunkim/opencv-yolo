import os
import numpy as np
import cv2

from opencv_yolo import initialize, predict, drawPred


def main():
    filename = "samples/image2.jpg"
    resize_ratio = 0.66

    img = cv2.imread(filename)
    img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)
    cv2.imshow("img", img)
    cv2.waitKey(1)

    # select yolo model
    YOLO_MODEL = "yolov3"
    YOLO_DATA = "coco"

    # initialize YOLO model
    net, classes, output_layers = initialize(YOLO_MODEL, YOLO_DATA)

    # predict objects
    height, width, channels = img.shape
    indexes, class_ids, confidences, boxes = predict(img, net, output_layers, (width, height))

    # draw result
    resImg = img.copy()
    drawPred(resImg, indexes, class_ids, confidences, boxes, classes)

    # write and display the result to a file
    cv2.imwrite("result.png", resImg)
    cv2.imshow("result", resImg)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    