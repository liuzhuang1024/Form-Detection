# Form-Detection
表格中的文本检测
****************
## 参数的设置
* preprocess
``````
def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    # 设置1,1,代表两个方向都进行滤波算子梯度计算
    # 可以只进行一个方向的梯度计算减少某个方向的线的数量
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=3)
    # 2. 二值化
    # 调整二值化的范围，比如195,255
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # 3. 膨胀和腐蚀操作的核函数
    # 更改膨胀腐蚀的内核，在某方向上膨胀或者腐蚀增强
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    # 可以控制迭代的次数，使得膨胀和腐蚀程度，以及去除噪声效果
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=1)

    # 7. 存储中间图片
    # 观察中间结果，查看哪一步进行的效果影响较大
    cv2.imwrite("binary.png", binary)
    cv2.imwrite("dilation.png", dilation)
    cv2.imwrite("erosion.png", erosion)
    cv2.imwrite("dilation2.png", dilation2)

    return dilation2
```````
* 去除面积较小的矩形，或者加入其他的决定
``````
def findTextRegion(img):
    region = []

    # 1. 查找轮廓
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # opencv3.4

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if area < 500:
            continue
``````