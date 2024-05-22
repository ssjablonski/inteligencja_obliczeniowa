import cv2
import numpy as np

# Read the image
image = cv2.imread('zdjecie.jpg')

# Convert to grayscale using the average method
gray_avg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to grayscale using the weighted method
r, g, b = cv2.split(image)
gray_weighted = 0.299*r + 0.587*g + 0.114*b
gray_weighted = np.round(gray_weighted).astype(np.uint8)

# Now you can compare the two grayscale images
cv2.imshow('Grayscale (Average)', gray_avg)
cv2.imshow('Grayscale (Weighted)', gray_weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()