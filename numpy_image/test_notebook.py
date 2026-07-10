import numpy as np
import matplotlib.image as mping

img = mping.imread('dog.jpeg')
print(img.shape)
print(img.dtype)
gray_img_01 = (np.max(img, axis=2) + np.min(img, axis=2)) / 2
print(gray_img_01[[0,0]])

gray_img_03 = np.mean(img, axis=2)
print("gray_img_03[[0, 0]] =", gray_img_03[[0, 0]])
