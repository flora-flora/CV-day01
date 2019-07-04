# coding: utf-8
import cv2
import random 
import numpy as np
from matplotlib import pyplot as plt

def combineDataEnrichment(imgPath):
    re_img=cv2.imread(imgPath)
    cv2.imwrite('1.jpg',re_img)
    
    def random_light_color(img):
    # brightness
        B,G,R = cv2.split(img)
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
        cv2.imwrite('2.jpg',img_merge)
        
    random_light_color(re_img)   
    img_crop = re_img[1:500,400:800]
    cv2.imwrite('3.jpg',img_crop)
    
    # rotation 旋转
    M = cv2.getRotationMatrix2D((re_img.shape[1] / 2, re_img.shape[0] / 2), 180, 0.3) # center, angle, scale
    img_rotate = cv2.warpAffine(re_img, M, (re_img.shape[1], re_img.shape[0]))
    cv2.imwrite('4.jpg', img_rotate)
    
    # Affine Transform
    rows, cols, ch = re_img.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(re_img, M, (cols, rows))
    cv2.imwrite('5.jpg', dst)
    
combineDataEnrichment('img/zixian.jpg')
#只输入图片地址



