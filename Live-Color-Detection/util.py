import cv2
import numpy as np

# live video capture
cap = cv2.VideoCapture(0)

while True:
    # ret checks if frame is successfully read, frame stores frame data
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    # terminating after keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# terminate the video
cap.release()
cv2.destroyAllWindows()