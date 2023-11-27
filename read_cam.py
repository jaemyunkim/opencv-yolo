import os
import numpy as np
import cv2


def main():
    cap = cv2.VideoCapture(1)   # you can try to insert the camera id from 0.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width} x {height}")
    
    # resize the camera resolution
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(f"Changed camera resolution: {width} x {height}")
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            count += 1
            print(f"frame {count}")
            
            frame = cv2.flip(frame, 1)  # horizontal flipping. If you want to vertical flipping insert 0.
            cv2.imshow("video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("done")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    