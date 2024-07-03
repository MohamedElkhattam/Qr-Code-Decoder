import cv2
from soln import *

img = cv2.imread('Test Cases/08-Compresso-Espresso.png',cv2.IMREAD_GRAYSCALE)
thresholded_image = Threshold_img(img)
closed_img = Open_Close_img_21(thresholded_image)
final_img = Component_Extraction(closed_img)[1]
QR_Print_BA(img , final_img)