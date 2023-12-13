import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


# set rows and cols
nrows = 3
ncols = 3   

# original image but in BRG
img = cv2.imread('ATU.jpg')

# original image in grey
gray = rgb2gray(img)

# original image in RGB
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(gray, cmap=plt.get_cmap('gray'))

# saves the image
plt.savefig('ATU_greyscale.png')

# blur image
blurImg = cv2.blur(gray, (5, 5))

# 13x13 averaging filter kernel
KernelSizeWidth = 13
KernelSizeHeight = 13
imgOut = cv2.GaussianBlur(blurImg, (KernelSizeWidth, KernelSizeHeight), 0)

# plot and dispaly images
plt.figure(1)

# Image 1 - Original image
plt.subplot(nrows, ncols, 1), plt.imshow(imgRGB, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

# Image 2 - Gray Scale image
plt.subplot(nrows, ncols, 2), plt.imshow(gray, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

# Image 3 - Blurred  3x3
plt.subplot(nrows, ncols, 3), plt.imshow(blurImg, cmap='gray')
plt.title('3 x 3 Blur'), plt.xticks([]), plt.yticks([])

# Image 4 - Blurred  13x13
plt.subplot(nrows, ncols, 4), plt.imshow(imgOut, cmap='gray')
plt.title('13 x 13 Blur'), plt.xticks([]), plt.yticks([])

# saves the image
plt.savefig('Sobel.png')

# Sobel Horizontal
sobelHorizontal = cv2.Sobel(imgOut, cv2.CV_64F, 1, 0, ksize=5)  # x dir
# Sobel Vertical
sobelVertical = cv2.Sobel(imgOut, cv2.CV_64F, 0, 1, ksize=5)  # y dir

# plt.figure(2)
# Image 5 - Sobel Horizontal
plt.subplot(nrows, ncols, 5), plt.imshow(sobelHorizontal, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

# Image 6 - Sobel Vertical
plt.subplot(nrows, ncols, 6), plt.imshow(sobelVertical, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

# plot sobelHorizontal and sobelVertical together
joinSobelH_SobelV = (sobelHorizontal + sobelVertical)

# Image 7 - sobelHorizontal + sobelVertical
plt.subplot(nrows, ncols, 7), plt.imshow(joinSobelH_SobelV, cmap='gray')
plt.title('Sobel Sum'), plt.xticks([]), plt.yticks([])

#  Canny edge detection
# lower threshold = 100
# upper threshold 200
edges = cv2.Canny(img, 100, 200, L2gradient=False)

# Image 8 - plot canny image
plt.subplot(nrows, ncols, 8), plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])

# deep copy of joinSobelH_SobelV
thresholdImage = joinSobelH_SobelV.copy()
thresh = 200
#height and width
height, width = thresholdImage.shape

# for loop to set all the values below the threshold to 0 and all above the threshold to 1
for i in range(0, height):
    for j in range(0, width):
        pix = thresholdImage[i, j]
        if pix < thresh:
            thresholdImage[i, j] = 0
        else:
            thresholdImage[i, j] = 1

# Image 10 - different thresholds
plt.subplot(nrows, ncols, 9), plt.imshow(thresholdImage, cmap='gray')
plt.title('ThresholdImage 13*13 - ' + str(thresh)
          ), plt.xticks([]), plt.yticks([])

plt.show()
