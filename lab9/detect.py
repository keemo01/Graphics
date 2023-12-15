import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('liberty.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform Harris corner detection
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Deep copy the original image for Harris corners
imgHarris = img.copy()

# Threshold for corner detection
threshold = 0.01

# Loop through and plot detected corners for Harris corners
for i in range(dst.shape[0]):
    for j in range(dst.shape[1]):
        if dst[i, j] > threshold * dst.max():
            cv2.circle(imgHarris, (j, i), 3, (0, 255, 0), -1)  # Adjust color (BGR)

# Display the image with Harris corners
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners')
plt.show()

# Perform Shi-Tomasi corner detection
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

corners = np.int0(corners)


# Deep copy the original image for Shi-Tomasi corners
imgShiTomasi = img.copy()

# Plot circles at detected corners for Shi-Tomasi corners
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(imgShiTomasi, (x, y), 3, (0, 0, 255), -1)  # Adjust color (BGR)

# Display the image with Shi-Tomasi corners
plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB))
plt.title('Shi-Tomasi Corners')
plt.show()

# Create ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors using ORB
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Draw keypoints on the image for ORB
img_ORB = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

# Display the image with ORB keypoints
plt.imshow(cv2.cvtColor(img_ORB, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoints')
plt.show()
