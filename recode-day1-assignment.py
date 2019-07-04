# coding: utf-8
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

img_zixian = cv2.imread('img/zixian.jpg')
#imread 第二个参数不写默认彩色图片，0表示以灰度模式读入图片
cv2.imshow('ouni',img_zixian)
cv2.waitKey()
#使显示停留，等待键盘输入，超时未输入 输出-1
cv2.imwrite('img/ouni.jpg',img_zixian)
#指定路径保存图片

#print(img_zixian)
print(img_zixian.shape)
B, G, R = cv2.split(img_zixian)
print(B.shape)
print(G.shape)
print(R.shape)
print(G)
cv2.imwrite('img/2.jpg',B)
cv2.imshow('B',B)
cv2.waitKey()

# change color 自定义的一个函数
def random_light_color(img):
    # brightness
    B, G, R = cv2.split(img)

    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))
    #img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img_merge

#函数调用
img_random_color = random_light_color(img_zixian)
cv2.imshow('img_random_color', img_random_color)
key = cv2.waitKey()
cv2.imwrite('img/1.jpg',img_random_color)

#img crop
img_crop = img_zixian[1:500,400:800]
cv2.imshow('img_crop',img_crop)
cv2.waitKey()

# gamma correction 也是改变亮度
img_dark = cv2.imread('img/zixian.jpg')
#cv2.imshow('img_dark', img_dark)
key = cv2.waitKey()
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(img_dark, table)
img_brighter = adjust_gamma(img_dark, 0.3)
cv2.imshow('img_dark', img_dark)
cv2.imshow('img_brighter', img_brighter)
key = cv2.waitKey()

# histogram
img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[0]*0.5), int(img_brighter.shape[1]*0.5)))
plt.hist(img_brighter.flatten(), 256, [0, 256], color = 'r')
img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])   # only for 1 channel
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)   # y: luminance(Ã÷ÁÁ¶È), u&v: É«¶È±¥ºÍ¶È
cv2.imshow('Color input image', img_small_brighter)
cv2.imshow('Histogram equalized', img_output)
key = cv2.waitKey(0)

# rotation 旋转
M = cv2.getRotationMatrix2D((img_zixian.shape[1] / 2, img_zixian.shape[0] / 2), 180, 0.3) # center, angle, scale
img_rotate = cv2.warpAffine(img_zixian, M, (img_zixian.shape[1], img_zixian.shape[0]))
cv2.imshow('rotated lenna', img_rotate)
key = cv2.waitKey(0)

print(M)

# set M[0][2] = M[1][2] = 0
print(M)

img_rotate2 = cv2.warpAffine(img_zixian, M, (img_zixian.shape[1], img_zixian.shape[0]))
cv2.imshow('rotated lenna2', img_rotate2)
key = cv2.waitKey(0)

# explain translation

# scale+rotation+translation = similarity transform
M = cv2.getRotationMatrix2D((img_zixian.shape[1] / 2, img_zixian.shape[0] / 2), 60, 0.5) # center, angle, scale
img_rotate = cv2.warpAffine(img_zixian, M, (img_zixian.shape[1], img_zixian.shape[0]))
cv2.imshow('rotated lenna', img_rotate)
key = cv2.waitKey(0)
print(M)

# Affine Transform
rows, cols, ch = img_zixian.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])
 
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img_zixian, M, (cols, rows))

cv2.imshow('affine', dst)
key = cv2.waitKey(0)

# perspective transform
def random_warp(img, row, col):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img_zixian, M_warp, (width, height))
    return M_warp, img_warp
M_warp, img_warp = random_warp(img_zixian, img_zixian.shape[0], img_zixian.shape[1])
cv2.imshow('lenna_warp', img_warp)
key = cv2.waitKey(0)

