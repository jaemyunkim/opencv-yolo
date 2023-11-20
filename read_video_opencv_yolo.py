import os
import numpy as np
import cv2

from utils import get_file_from_url
from opencv_yolo import opencv_yolo


def main():
    # video path
    video_urls = [
        "https://motchallenge.net/sequenceVideos/MOT16-07-raw.mp4", # MOT17 tracking video
    ]
    filenames = get_file_from_url(video_urls, "samples")

    # video preparation
    cap = cv2.VideoCapture(str(filenames[0]))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    resize_ratio = 0.66
    
    # select yolo model
    YOLO_MODEL = "yolov3"
    YOLO_MODEL = "yolov5s"
    YOLO_DATA = "coco"

    # initialize YOLO model
    od = opencv_yolo(YOLO_MODEL, YOLO_DATA)

    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            count += 1
            print(f"frame {count}")
            frame = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio)
            cv2.imshow("video", frame)

            # predict objects
            od.predict(frame)

            # draw result
            resImg = frame.copy()
            od.drawPred(resImg)

            cv2.imshow("result", resImg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("done")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    