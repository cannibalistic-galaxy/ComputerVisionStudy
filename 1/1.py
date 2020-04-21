import cv2

if __name__ == '__main__':
    img = cv2.imread('E:/PythonProgram/opencv_study/fig_transaction/yoona.jpg', cv2.IMREAD_UNCHANGED)
    print('Original Dimensions:', img.shape)
    scale_percent = 30
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    fx = 1.5
    fy = 1.5
    resized1 = cv2.resize(resized, dsize=None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    resized2 = cv2.resize(resized, dsize=None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    print('Resized Dimensions:', resized.shape)
    cv2.imshow("yuantu", img)
    cv2.imshow("Resized image", resized)
    cv2.imshow("INTER_NEAREST image", resized1)
    cv2.imshow("INTER_LINEAR image", resized2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()