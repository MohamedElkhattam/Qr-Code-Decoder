import cv2 
import numpy as np
from soln import *

img = cv2.imread('Test Cases/14-BANANAAA!!!.png' ,cv2.IMREAD_GRAYSCALE)
matrix1 = np.float32([[576,563], [694,518],[620,680],[736,637]])
New_prespective_qr = perspective_transform_and_warp(matrix1 , img)
final_img = Threshold_img(New_prespective_qr)

#Will be added to the pipline at the end of the pipeline
square_se=cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
new_final = cv2.dilate(final_img , square_se)
QR_Print_BA(img , final_img)