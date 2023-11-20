import os
import numpy as np
import cv2


def main():
    filename = "samples/image1.jpg"
    resize_ratio = 0.66

    img = cv2.imread(filename)
    img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    