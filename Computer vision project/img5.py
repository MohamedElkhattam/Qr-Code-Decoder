import cv2
import numpy as np
from soln import *
from main import *
img = cv2.imread('Test Cases/05-Caesar-cipher.png',cv2.IMREAD_GRAYSCALE)
thresholded_image = Threshold_img(img)
final_img = Open_Close_img_21(thresholded_image)
image_decode(final_img)
QR_Print_BA(img , final_img)