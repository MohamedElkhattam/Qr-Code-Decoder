import cv2
from soln import *

img = cv2.imread("Test cases/01-Getting-started.png" ,cv2.IMREAD_GRAYSCALE)
final_img = Threshold_img(img)
QR_Print_BA(img , final_img)
