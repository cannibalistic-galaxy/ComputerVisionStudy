import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('E:/PythonProgram/opencv_study/fig_transaction/opencv.PNG')
b, g, r = cv2.split(img)
img = cv2.merge([r, g, b]) # 这两行代码解决rgb排列的问题

# 1、均值滤波
blur = cv2.blur(img, (5,5))

# 2、高斯滤波
gblur = cv2.GaussianBlur(img,(5,5),0)

# 3、中值滤波
median = cv2.medianBlur(img, 5)

#4、盒子滤波
box = cv2.boxFilter(img, -1, (5,5))

plt.subplot(231),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(gblur),plt.title('GaussianBlurred')
plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(median),plt.title('MedianBlurred')
plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(box),plt.title('BoxBlurred')
plt.xticks([]), plt.yticks([])
plt.show()