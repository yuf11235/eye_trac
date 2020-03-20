# intergrate dlib_image_test.py

import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt

# 加载算法模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('etc/shape_predictor_68_face_landmarks.dat')

# 读取图像文件
img = cv2.imread("data\\beauty.jpg")
print(img.shape)
fy = round((480 / img.shape[0]), 2)
fx = round((640 / img.shape[1]), 2)
img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 识别人脸区域
rects = detector(gray, 1)
if len(rects) > 0:
    rect = rects[0]
    # 识别人脸特征点
    landmarks = np.array([[p.x, p.y] for p in predictor(img, rect).parts()])
    # 标记人脸区域和特征点
    img = cv2.rectangle(img, (rect.left(), rect.top()),
                        (rect.right(), rect.bottom()), (255, 255, 255))
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        cv2.circle(img, pos, 1, color=(0, 255, 0))
    # 标记眼部位置，框出眼睛位置
    eye_list = landmarks[36: 42]
    o_x = min(eye_list[:, 0])
    o_y = min(eye_list[:, 1])
    eye_right_down_x = max(eye_list[:, 0])
    eye_right_down_y = max(eye_list[:, 1])
    img = cv2.rectangle(img, (o_x, o_y), 
                        (eye_right_down_x, eye_right_down_y), (255, 255, 255))
    left_eye = gray[o_y: eye_right_down_y, o_x: eye_right_down_x]
    # cv2.imshow('left eye', left_eye)
    # 转换为二值图
    maxi = float(left_eye.max())
    mini = float(left_eye.min())
    x = maxi - ((maxi - mini) // 1.4)
    # 二值化,返回阈值ret  和  二值化操作后的图像thresh
    ret, thresh = cv2.threshold(left_eye, x, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary', thresh)
    # 用数学形态学处理一下
    # 消除小的区域，保留大块的区域，从而定位车牌
    # 进行闭运算
    kernel = np.ones((3, 2), np.uint8)
    closing_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closing_img", closing_img)
    # 进行开运算
    kernel = np.ones((2, 2), np.uint8)
    opening_img = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel)
    cv2.imshow("opening_img_1", opening_img)
    # # 进行闭运算
    # kernel = np.ones((4, 1), np.uint8)
    # closing_img = cv2.morphologyEx(opening_img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closing_img_2", closing_img)
    # binary_eye = closing_img_2
    # kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # kernel_dilated = cv2.dilate(closing_img, kernel_2)
    # cv2.imshow('kernel_dilated', kernel_dilated)
    # 求出眼球的边缘，使用canny检测
    # cannyimg = cv2.Canny(closing_img, closing_img.shape[0], closing_img.shape[1])
    # cv2.imshow("cannyimg", thresh)

    # 求出瞳仁位置及其相对于直视时瞳孔的位置，用重心法
    # binary_eye = binary_eye == 0
    binary_eye = opening_img == 0
    x_c = np.sum(np.dot(binary_eye, np.arange(
        binary_eye.shape[1]))) / np.sum(binary_eye)
    y_c = np.sum(np.dot(binary_eye.T, np.arange(
        binary_eye.shape[0]))) / np.sum(binary_eye)
    print("(x_c, y_c) = ({}, {})".format(x_c, y_c))
    # 不妨设直视时，瞳孔的为值为37、38、40、41几个点的平均值
    x_orth, y_orth = np.sum(eye_list.reshape(-1, 2), axis=0) / 6 - [o_x, o_y]
    print("(x_orth, y_orth) = ({}, {})".format(x_orth, y_orth))
    # 判断眼睛向左向右
    if x_c <= x_orth*0.8:
        h_direction = '右'
    elif x_c <= x_orth*1.2:
        h_direction = '中'
    else:
        h_direction = '左'
    print(h_direction)
    # 判断眼睛向上向下
    if y_c <= y_orth*0.75:
        v_direction = '上'
    elif y_c <= y_orth*1.25:
        v_direction = '中'
    else:
        v_direction = '下'
    print(v_direction)

    # plt.scatter(landmarks[:, 0], -landmarks[:, 1])
    # plt.scatter(left_eye[:, 0], -left_eye[:, 1])
    # plt.grid()
    # plt.show()
cv2.imshow('landmarks', img)
# 释放资源
cv2.waitKey()
cv2.destroyAllWindows()
