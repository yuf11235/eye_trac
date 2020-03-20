import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt

# 加载算法模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('etc/shape_predictor_68_face_landmarks.dat')

# 读取图像文件
img = cv2.imread("data\\beauty.jpg")
# img = cv2.imread("data\girl.jpg")
# img = cv2.imread("data\\timg.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 识别人脸区域
rects = detector(gray, 1)
if len(rects) > 0:
    rect = rects[0]
    # 识别人脸特征点
    landmarks = np.array([[p.x, p.y] for p in predictor(img, rect).parts()])
    # 标记人脸区域和特征点
    img = cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 255, 255))
    for idx, point in enumerate(landmarks):
        pos = (point[0], point[1])
        # 计算两个端点到中心的二值图的像素值的和，哪边大即为眼睛的方向
        #
        print("pos[{}]: {}".format(idx, pos))
        cv2.circle(img, pos, 1, color=(0, 255, 0))
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(idx + 1), pos, font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    # 标记眼部位置，框出眼睛位置
    img = cv2.rectangle(img, (landmarks[36][0], min(landmarks[37][1], landmarks[38][1])), 
                        (landmarks[39][0], max(landmarks[41][1], landmarks[42][1])), (255, 255, 255))
    # 将左眼单独取出来做形态学分析
    o_x = landmarks[36][0]
    o_y = min(landmarks[37][1], landmarks[38][1])
    left_eye = gray[min(landmarks[37][1], landmarks[38][1]): max(
        landmarks[41][1], landmarks[42][1]), landmarks[36][0]:landmarks[39][0]]
    # print(left_eye)
    # cv2.imwrite('calibresult.png', dst)
    cv2.imshow('left eye', left_eye)
    # 转换为二值图
    maxi = float(left_eye.max())
    mini = float(left_eye.min())
    x = maxi - ((maxi - mini) // 1.4)
    # 二值化,返回阈值ret  和  二值化操作后的图像thresh
    ret, thresh = cv2.threshold(left_eye, x, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary', thresh)
    print("left_eye binary")
    print(type(thresh))

    # 用数学形态学处理一下
    # 消除小的区域，保留大块的区域，从而定位车牌
    # 进行闭运算
    kernel = np.ones((2, 1), np.uint8)
    closing_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("closing_img", closing_img)
    # 进行开运算
    opening_img = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel)
    cv2.imshow("opening_img_1", opening_img)

    binary_eye = opening_img

    # 求出瞳仁位置及其相对于直视时瞳孔的位置，用重心法
    binary_eye = binary_eye == 0
    # print(binary_eye)
    # for i in range(binary_eye.shape[0]):
    #     for j in range(binary_eye.shape[1]):
    x_c = np.sum(np.dot(binary_eye, np.arange(binary_eye.shape[1]))) / np.sum(binary_eye)
    y_c = np.sum(np.dot(binary_eye.T, np.arange(binary_eye.shape[0]))) / np.sum(binary_eye)
    print("(x_c, y_c) = ({}, {})".format(x_c, y_c))
    # 不妨设直视时，瞳孔的为值为37、38、40、41几个点的平均值
    x_orth, y_orth = (landmarks[37] + landmarks[38] +
              landmarks[40] + landmarks[41]) / 4 - [o_x, o_y]
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
