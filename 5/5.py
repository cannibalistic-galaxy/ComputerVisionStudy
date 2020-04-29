import cv2
import numpy as np
import matplotlib.pyplot as plt

# 一、简单阈值法
img = cv2.imread('E:/PythonProgram/opencv_study/fig_transaction/light.PNG',0)
ret , thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret , thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret , thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret , thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret , thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['original image','Binary','binary-inv','trunc','tozero','tozero-inv']
images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
# 二、大津法
# ret , thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# b, g, r = cv2.split(img)
# img = cv2.merge([r, g, b])
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret1, th1 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
ret2, th2 = cv2.threshold(img,255,255,cv2.THRESH_OTSU)
imgs = np.hstack([img, th1, th2])
cv2.imshow('OTSU', imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.subplot(121), plt.imshow(th1)
# plt.subplot(122), plt.imshow(th2)
# plt.xticks([]),plt.yticks([])
# plt.show()
# cv2.imshow('OTSU',th1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('OTSU',th2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 三、自适应阈值法
#中值滤波
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.medianBlur(img,5)
ret , th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# 11为block size，2为C值
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY,11,2)
imgs = np.hstack([img, th1, th2, th3])
cv2.imshow('OTSU', imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()