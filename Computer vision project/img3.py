import cv2
from soln import *

img = cv2.imread("Test cases/03-Leffy-bina-ya-donya.png", cv2.IMREAD_GRAYSCALE)

Vertical_Flipped_img   = flip_image(img , 1)
Hortizonta_Flipped_img = flip_image(Vertical_Flipped_img, 0)
final_img  = Threshold_img(Hortizonta_Flipped_img)
QR_Print_BA(img , final_img)
