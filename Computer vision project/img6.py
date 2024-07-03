import cv2
import numpy as np
from soln import *


img = cv2.imread('Test Cases/06-Railfence-cipher.png' ,cv2.IMREAD_GRAYSCALE)
matrix1 = np.float32([[316,198], [974,93],[61,950],[718,846]])
Perspective_Image = perspective_transform_and_warp(matrix1 , img)
threholded_img = Threshold_img(Perspective_Image)
final_img = Open_Close_img_21(threholded_img)
QR_Print_BA(img , final_img)