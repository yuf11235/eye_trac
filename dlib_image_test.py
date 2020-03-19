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
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

"""   # 先定义一个元素结构
r = 16
h = w = r * 2 + 1
kernel = np.zeros((h, w), dtype=np.uint8)
cv2.circle(kernel, (r, r), r, 1, -1)
# 开运算,去噪声
openingimg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
cv2.imshow("openingimg_1", openingimg)
# 获取差分图
strtimg = cv2.absdiff(gray, openingimg)
cv2.imshow("strtimg", strtimg)

# 转换为二值图
maxi = float(img.max())
mini = float(img.min())
x = maxi - ((maxi - mini) / 2)
# 二值化,返回阈值ret  和  二值化操作后的图像thresh
ret, thresh = cv2.threshold(strtimg, x, 255, cv2.THRESH_BINARY)
cv2.imshow('binary', thresh)"""

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
        print(pos)
        cv2.circle(img, pos, 1, color=(0, 255, 0))
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(idx + 1), pos, font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    plt.scatter(landmarks[:, 0], -landmarks[:, 1])
    plt.show()
cv2.imshow('landmarks', img)
# 释放资源
cv2.waitKey()
cv2.destroyAllWindows()
