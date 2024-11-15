#importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

#importing image
im_path = "C:/Users/User/Documents/Git/My-Computer-Vision-Projects/Document-Scanner/IMG_2714.jpg"
img = cv2.imread(im_path)
original = img.copy()
img = cv2.resize(img, (800, 1500))

'''Remove Noise (Anything outside the receipt) from image'''
# Turning image into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blurring Image
blurred = cv2.GaussianBlur(gray, (7,7), 0)

# Making the image back to BGR
regen = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

#Edge Detection
edge_detected = cv2.Canny(regen, 50, 150)

# Contour Extraction
contoured,_ = cv2.findContours(edge_detected, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

contoured = sorted(contoured, reverse=True, key=cv2.contourArea)

#Select the best contours
for c in contoured:
    epsilon = 0.02 * cv2.arcLength (c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    if len(approx) == 4:
        target = approx
        break


def reorder(points):
    points = points.reshape((4, 2))
    ordered = np.zeros((4, 2), dtype=np.float32)

    add = points.sum(axis=1)
    ordered[0] = points[np.argmin(add)]  # Top-left
    ordered[2] = points[np.argmax(add)]  # Bottom-right

    diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(diff)]  # Top-right
    ordered[3] = points[np.argmax(diff)]  # Bottom-left
    
    return ordered

reordered_contour = reorder(target)
print("Reordered points:", reordered_contour)



input_rep = reordered_contour
out_rep = np.float32([[0, 0], [800, 0], [800, 1500], [0, 1500]])

M = cv2.getPerspectiveTransform (input_rep, out_rep)
ans = cv2.warpPerspective (img, M, (800, 1500))
plt.subplot(141), plt.imshow(gray, cmap='gray'), plt.title('Grayscale')
plt.subplot(142), plt.imshow(blurred, cmap='gray'), plt.title('Blurred')
plt.subplot(143), plt.imshow(edge_detected, cmap='gray'), plt.title('Edge Detected')
plt.subplot(144), plt.imshow(ans, cmap='gray'), plt.title('Final')
plt.show()

