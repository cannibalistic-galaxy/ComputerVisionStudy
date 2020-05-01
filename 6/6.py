import cv2
import numpy as np
import matplotlib.pyplot as plt

def imgshow(name, source, number=0):
    if number != 0:
        plt.subplot(number)
    plt.imshow(source, cmap='gray')
    plt.title(str(name)), plt.xticks([]), plt.yticks([])

img = cv2.imread('E:/PythonProgram/opencv_study/fig_transaction/qipan.jpg', 0)

# 1、sobel算子
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5) # 64F代表每一个像素点元素占64位浮点数,通道数为1
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# 2、拉普拉斯算子
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# 3、canny算子
gaussian = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(gaussian, 50, 150)
canny2 = cv2.Canny(gaussian, 10, 20)

imgshow('Original', img, 231)
imgshow('Sobelx', sobelx, 232)
imgshow('Sobely', sobely, 233)
imgshow('Laplacian', laplacian, 234)
imgshow('Canny_50-150', canny, 235)
imgshow('Canny_10-20', canny2, 236)
plt.show()