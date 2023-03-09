# -*-coding:utf-8 -*-

"""
# File       : hough.py
# Time       ：2023/3/8 15:27
# Author     ：sz
# version    ：python 3.9
# Description：
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def hough_line(img, step_theta=6, step_r=1):
    rows, cols = img.shape
    l = (rows ** 2 + cols ** 2) ** (1 / 2)
    num_theta = int(180 / step_theta)
    num_r = int(np.ceil(2 * l / step_r))
    accumulator = np.zeros([num_r, num_theta])  # 创建累加器

    sin_theta = [np.sin(t * np.pi / num_theta) for t in range(num_theta)]
    cos_theta = [np.cos(t * np.pi / num_theta) for t in range(num_theta)]

    draw_dic = {}
    for i in range(num_r):
        for j in range(num_theta):
            draw_dic[(i, j)] = []

    for i in range(rows):
        for j in range(cols):
            if img[i][j] == 255:
                for k in range(num_theta):
                    r = j * cos_theta[k] + i * sin_theta[k]
                    num_r_loc = int(round(r + l) / step_r)
                    accumulator[num_r_loc][k] += 1
                    draw_dic[(num_r_loc, k)].append((j, i))

    return accumulator, draw_dic


if __name__ == '__main__':
    img = cv2.imread('02.jpeg')
    print(img.shape)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 50, 150)
    accmulator, draw_dic = hough_line(edge)
    for i in range(accmulator.shape[0]):
        for j in range(accmulator.shape[1]):
            if accmulator[i][j] >= 40:
                points = draw_dic[(i, j)]
                cv2.line(img, points[0], points[-1], (255, 0, 0), 2)
    plt.imshow(img, cmap="gray")
    plt.show()





