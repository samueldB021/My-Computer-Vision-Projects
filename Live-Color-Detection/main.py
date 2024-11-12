import cv2
import numpy as np
from PIL import Image
from util import get_limits

blue = [255, 0, 0] #in BGR colorspace
lowerLimit, upperLimit = get_limits(color=blue)
# live video capture
cap = cv2.VideoCapture(0)

while True:
    # ret checks if frame is successfully read, frame stores frame data
    ret, frame = cap.read()

    # convert the image into HSV
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # highlight all pixels in the range attained
    mask = cv2.inRange(hsvImg, lowerLimit, upperLimit)
    
    # getting bounding box to draw rectangles around objects
    reversemask = Image.fromarray(mask)
    bbox = reversemask.getbbox()

    # drawing a rectangle over the colour
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    # terminating after keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# terminate the video
cap.release()
cv2.destroyAllWindows()