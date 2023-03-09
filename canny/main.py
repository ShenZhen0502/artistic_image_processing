# -*-coding:utf-8 -*-

"""
# File       : main.py
# Time       ：2023/2/19 14:58
# Author     ：sz
# version    ：python 3.9
# Description：
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
import cv2
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def convolve(image, kernel, padding, stride):
    """
    :param image: 要卷积的图像
    :param kernel: 卷积核
    :param padding: 输入要上下左右四个方向要padding的宽度，例如[1,1,1,1]代表四个方向padding宽度为1
    :param stride: 代表卷积的步长
    :return: 卷积后的图像
    """
    result = None  # 保存最后的结果，先在函数域声明一下
    img_size = image.shape  # 输入图像的形状
    filter_size = kernel.shape  # 滤波核的形状，传统方法不同于深度学习，滤波核是二维的，深度学习卷积核一般是三维的

    if len(img_size) == 3:  # 判断输入图像维度，如果是三维的，一般指彩色图像
        result = []  # 用来存储卷积之后的结果
        for i in range(0, img_size[-1]):  # 遍历通道
            channel = []  # 创建一个通道列表，用来存储通道数据
            padded_img = np.pad(image[:, :, i], ((padding[0], padding[1]), (padding[2], padding[3])),
                                'constant')  # 对每个通道进行padding
            # 遍历行维度，添加一个列表到channel尾部，用来存放生成数据
            for j in range(0, img_size[0], stride):  # 遍历行
                channel.append([])  # 添加一个空列表到通道列表中，空列表用来存储卷积后的每一行的数据
                for k in range(0, img_size[1], stride):  # 遍历列
                    val = (kernel * padded_img[j:j + filter_size[0], k:k + filter_size[1]]).sum()  # 将滤波核与遍历图像相乘
                    channel[-1].append(val)  # 将得到的数据添加到刚刚的空列表中
            result.append(channel)  # 最后将通道列表添加到最后的结果中

    elif len(img_size) == 2:  # 如果输入图像是二维的，一般指灰度图或二值图
        result = []  # 存储卷积之后的结果
        padded_img = np.pad(image, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')  # 进行padding
        for j in range(0, img_size[0], stride):  # 遍历行
            result.append([])  # 因为图像是二维的，就不像三维的需要一个通道维度，我们直接添加一个空列表，作为输出结果的行
            for k in range(0, img_size[-1], stride):  # 遍历列
                val = (kernel * padded_img[j:j + filter_size[0], k:k + filter_size[1]]).sum()  # 计算卷积结果
                result[-1].append(val)  # 将结果添加到行中

    return result


def im2col(image, filter_h, filter_w, padding, stride):
    """
    只支持二维图像，传统方法一般都是用灰度图或二值图
    :param image: 要转换的图像
    :param filter_h: 滤波的长
    :param filter_w: 滤波的宽
    :param padding: padding的宽度
    :param stride: 卷积的步长
    :return: 要转换的图像
    """

    matrix = []  # 保存最后的输出结果
    img_shape = image.shape  # 读取输入图像形状
    image = np.pad(image, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')  # 对输入图像进行pad
    for j in range(0, img_shape[0], stride):  # 遍历行
        for k in range(0, img_shape[1], stride):  # 遍历列
            col = image[j:j + filter_w, k:k + filter_h].reshape(1, -1)  # 找到对应位置的数据（二维的），将其reshape成一行数据
            matrix.append(col)  # 将reshape的数据添加到结果列表中

    return np.array(matrix)


def non_maximum_suppress(grad_l, grad_d):
    """
    :param grad_l: 图像的梯度大小
    :param grad_d: 图像的梯度方向
    :return: 抑制之后图像的梯度大小
    """
    h = grad_l.shape[1]  # 获取图像的行数
    w = grad_l.shape[0]  # 获取图像的列数
    result = np.zeros((h, w))  # 创建和图像同样形状的全零矩阵，用来存放结果数据

    for i in range(1, h - 2):  # 遍历行
        for j in range(1, w - 2):  # 遍历列

            eight_neibor = grad_l[i - 1:i + 2, j - 1:j + 2]  # 取出图像梯度大小的八领域
            if 0 <= grad_d[i][j] <= 1:  # 如果梯度大小为[0,1]之间，即角度在0-45和180-225
                dTmp1 = (1 - grad_d[i][j]) * eight_neibor[1][2] + grad_d[i][j] * eight_neibor[0][2]  # 按照公式计算dTmp1
                dTmp2 = (1 - grad_d[i][j]) * eight_neibor[2][0] + grad_d[i][j] * eight_neibor[1][0]  # 按照公式计算dTmp2
                if grad_l[i][j] > dTmp1 and grad_l[i][j] > dTmp2:  # 如果这个中心点比dTmp1和dTmp2都大，那么这个点就是极大值点，
                    # 我们就把它放到结果矩阵对应的【i，j】位置
                    result[i][j] = grad_l[i][j]  # 放到结果矩阵中
            if grad_d[i][j] > 1:  # 角度在45-90和225-270
                dTmp1 = (1 - 1 / grad_d[i][j]) * eight_neibor[0][1] + 1 / grad_d[i][j] * eight_neibor[0][2]
                dTmp2 = (1 - 1 / grad_d[i][j]) * eight_neibor[2][1] + 1 / grad_d[i][j] * eight_neibor[2][0]
                if grad_l[i][j] > dTmp1 and grad_l[i][j] > dTmp2:
                    result[i][j] = grad_l[i][j]
            if grad_d[i][j] < -1:  # 角度在90-135和270-315
                dTmp1 = (1 - 1 / grad_d[i][j]) * eight_neibor[0][1] + 1 / grad_d[i][j] * eight_neibor[0][0]
                dTmp2 = (1 - 1 / grad_d[i][j]) * eight_neibor[2][1] + 1 / grad_d[i][j] * eight_neibor[2][2]
                if grad_l[i][j] > dTmp1 and grad_l[i][j] > dTmp2:
                    result[i][j] = grad_l[i][j]
            if -1 <= grad_d[i][j] <= 0:  # 角度在135-180和315-360
                dTmp1 = (1 - grad_d[i][j]) * eight_neibor[1][0] + grad_d[i][j] * eight_neibor[0][0]
                dTmp2 = (1 - grad_d[i][j]) * eight_neibor[1][2] + grad_d[i][j] * eight_neibor[2][2]
                if grad_l[i][j] > dTmp1 and grad_l[i][j] > dTmp2:
                    result[i][j] = grad_l[i][j]

    return result  # 返回最后的结果


def twothrehold(inp, low, high):
    """
    :param inp: 输入要双阈值抑制的图像
    :param high: 高阈值，
    :param low: 低阈值，
    :return:
    """
    inp[inp >= high] = 255  # 比这个阈值大的一定是边缘点，都记为255
    inp[inp <= low] = 0  # 比这个阈值低的一定不是边缘点，都记为0

    return inp


def connect(inp, low, high):
    """
    对处于阈值之间的像素点进行进一步判断，如果该点旁边有一个肯定是边缘点的，那么我们就将这个点视为边缘点，这样就把断断续续的边缘点连接起来
    了。具体怎么做呢？利用栈，把通过双阈值的已经确定为边缘点的点全部压入栈中，然后弹栈，将弹出的元素的八邻域中在双阈值之间的点视为边缘点，
    再将这些点压入栈中。再弹栈，重复操作，直至栈空。
    :param inp:  图像梯度大小
    :param low:  高阈值
    :param high: 低阈值
    :return:
    """
    st = []  # stack 栈
    img_shape = inp.shape
    cood = []  # 存放像素点坐标

    for i in range(1, img_shape[0]):  # 遍历行
        cood.append([])  # 图像是二维的，所以该空列表用来存放行的坐标
        for j in range(1, img_shape[1]):  # 遍历列
            cood[-1].append([i, j])  # 将坐标添加到cood中
            if inp[i][j] == 255:  # 如果这个梯度值为255，那么它一定是边缘点，那么将其入栈
                st.append([i, j])  # 入栈
    cood = np.array(cood)

    while len(st) != 0:  # 判断栈是否为空
        d = st.pop()  # 弹出栈尾元素
        cood_eight_neibor = cood[d[0] - 1:d[0] + 2, d[1] - 1:d[1] + 2, :]  # 找到这个元素的八领域坐标

        for i in range(3):
            for j in range(3):
                if low < inp[cood_eight_neibor[i][j][0], cood_eight_neibor[i][j][1]] < high:  # 如果八领域内梯度值在阈值之间
                    inp[cood_eight_neibor[i][j][0], cood_eight_neibor[i][j][1]] = 255  # 一定是边缘点，记为255
                    st.append(list(cood_eight_neibor[i][j]))  # 再将其入栈

    inp[inp != 255] = 0  # 剩下的点都不是边缘点，记为0

    return inp


def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    mid = size // 2
    for i in range(size):
        for j in range(size):
            kernel[i][j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((i - mid) ** 2 + (j - mid) ** 2) / (2 * sigma ** 2))

    return kernel / np.sum(kernel)


if __name__ == '__main__':
    img = plt.imread(r"C:\Users\sz\Desktop\study\传统图像处理\artistic image processing\lena.png")
    img = np.dot(img, [0.299, 0.587, 0.114])
    print(img.shape)
    gaussian_k = gaussian_kernel(3, 0.8)
    img = np.array(convolve(img, gaussian_k, [1, 1, 1, 1], 1))
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_img_y = convolve(img * 255, sobel_kernel_y, [1, 1, 1, 1], 1)
    sobel_img_y = np.array(sobel_img_y)
    times = time.time()
    sobel_img_x = convolve(img * 255, sobel_kernel_x, [1, 1, 1, 1], 1)
    timess = time.time()
    print(timess - times)
    sobel_img_x = np.array(sobel_img_x)
    # plt.figure()
    # plt.subplot(2, 2, 2)
    # plt.imshow(sobel_img_y.astype(np.uint8), cmap="gray")
    # plt.title('普通卷积搭配sobel_kernel_y')
    # plt.subplot(2, 2, 1)
    # plt.imshow(sobel_img_x.astype(np.uint8), cmap="gray")
    # plt.title('普通卷积搭配sobel_kernel_x')
    # plt.axis('off')
    # times = time.time()
    # col_img = im2col(img, 3, 3, [1, 1, 1, 1], 1)
    # sobel_x = sobel_kernel_x.reshape(-1, 1)
    # outs = np.dot(col_img*255, sobel_x).reshape(512, 512)
    # timess = time.time()
    # print(timess-times)
    # plt.subplot(2, 2, 3)
    # plt.imshow(outs.astype(np.uint8), cmap="gray")
    # plt.title('im2col搭配sobel_kernel_x')
    #
    # sobel_y = sobel_kernel_y.reshape(-1, 1)
    # outs = np.dot(col_img*255, sobel_y).reshape(512, 512)
    # plt.subplot(2, 2, 4)
    # plt.imshow(outs.astype(np.uint8), cmap="gray")
    # plt.title('im2col搭配sobel_kernel_y')
    # plt.show()

    grad_l = (sobel_img_x ** 2 + sobel_img_y ** 2) ** 0.5
    sobel_img_x[sobel_img_x == 0] = 0.0000000001
    grad_t = sobel_img_y / sobel_img_x
    nms = non_maximum_suppress(grad_l, grad_t)
    print(nms.shape)
    plt.subplot(2, 2, 1)
    plt.title("梯度图")
    plt.imshow(grad_l, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.title("非极大值抑制")
    plt.imshow(nms, cmap="gray")
    plt.subplot(2, 2, 3)
    plt.title("双阈值处理")
    nms_out = twothrehold(nms, 30, 90)
    plt.imshow(nms_out, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.title("最终边缘检测图")
    result = connect(nms_out, 30, 90)
    plt.imshow(result, cmap="gray")
    plt.show()

    i = cv2.imread("./lena.png")
    out = cv2.Canny(i, 100, 250)
    plt.imshow(out, cmap="gray")
    plt.show()
