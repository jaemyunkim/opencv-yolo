import os
import numpy as np
import cv2

from opencv_yolo import opencv_yolo


def main():
    filename = "samples/image1.jpg"
    resize_ratio = 0.66

    img = cv2.imread(filename)
    img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)
    cv2.imshow("img", img)
    cv2.waitKey(1)

    # select yolo model
    YOLO_MODEL = "yolov5s"
    YOLO_DATA = "coco"

    # initialize YOLO model
    od = opencv_yolo(YOLO_MODEL, YOLO_DATA)

    # predict objects
    od.predict(img)

    # draw result
    resImg = img.copy()
    od.drawPred(resImg)

    # write and display the result to a file
    cv2.imwrite("result.png", resImg)
    cv2.imshow("result", resImg)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    