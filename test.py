import cv2
import numpy as np


# read the image
img = cv2.imread("images/girl.jpg")
print(img[1, 0]) # BGR

#     img[col,row]