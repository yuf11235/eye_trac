import numpy as np
import cv2
import dlib

# 加载算法模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('etc/shape_predictor_5_face_landmarks.dat')

# 相机资源
cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 采集循环
while cap.isOpened():
    ret, img = cap.read()
    if ret:
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

        
        # 显示图像
        cv2.imshow('Camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# 释放资源
cap.release()
cv2.destroyAllWindows()
