import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(name, img):
    cv2.imshow(str(name), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('E:/PythonProgram/opencv_study/fig_transaction/yoona.jpg', cv2.IMREAD_UNCHANGED)
rows, cols, channel = img.shape
# 1、平移操作
a = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img, a, (cols, rows))
show('img', dst)
# 2、翻转操作
img_flip = cv2.flip(img, 0)
show('flip', img_flip)
# 3、旋转操作
rst = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 0.6)
dst = cv2.warpAffine(img, rst, (cols, rows))
show('rotation', dst)
# 4、仿射变换  图像的旋转加上拉升就是图像仿射变换
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols,rows))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()  # 直接使用plt输出的话会颜色差别很大，因为opencv的接口使用BGR，而matplotlib.pyplot 则是RGB模式
"""
代码cv2读入的是BGR模式，在opencv里面存储的是BGR，所以img用opencv输出就是正常颜色；
而matplotlib.pyplot是RGB模式，当用cv读入，直接用matplotlib.pyplot输出，颜色就变了，所以需要调整颜色的顺序，就变成了img2；
"""
# 下面的代码解决了这个问题
b, g, r = cv2.split(img)
img2 = cv2.merge([r, g, b])
pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img2, M, (cols,rows))
plt.subplot(121),plt.imshow(img2),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
# 5、透视变换
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M=cv2.getPerspectiveTransform(pts1,pts2)
dst=cv2.warpPerspective(img2,M,(300,300))
plt.subplot(121),plt.imshow(img2),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()