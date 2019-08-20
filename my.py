# import sys
# sys.path.append('/usr/local/lib/python2.7/dist-packages/')

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('0.jpg')
mser = cv2.MSER_create(_min_area=300)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
gray = cv2.dilate(gray, kernel1)
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
gray = cv2.erode(gray, kernel1)
regions, boxes = mser.detectRegions(gray)

for box in boxes:
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# plt.imshow(img, 'brg')
# plt.show()
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()