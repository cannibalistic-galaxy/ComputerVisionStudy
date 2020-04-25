import cv2
import numpy as np

def rgb2gray_mean(img):
    # ratio = 1.0 / 3
    # 转换类型
    int_img = img.astype(np.int32)
    print(int_img.shape)
    result = 0.229*int_img[...,0] + 0.587*int_img[...,1] + 0.114*int_img[...,2]
    return result.astype(np.uint8)

# 程序入口
def main():
    # 读取lena图
    color = cv2.imread('E:/PythonProgram/opencv_study/fig_transaction/yoona.jpg')
    # 转灰度
    gray = rgb2gray_mean(color)
    gray2 = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    # 显示
    cv2.imshow('color', color)
    cv2.imshow('gray', gray)
    cv2.imshow('gray2', gray2)
    cv2.imshow('hsv', hsv)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()