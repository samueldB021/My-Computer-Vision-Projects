# getting upper and lower limits of a color in HSV space
import cv2
import numpy as np

def get_limits (color):
    col = np.uint8([[color]]) #insert bgr value to be converted to hsv
    hsvCol = cv2.cvtColor (col, cv2.COLOR_BGR2HSV)

    lowerLimit = hsvCol[0][0][0] - 10, 100, 100
    upperLimit = hsvCol[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit