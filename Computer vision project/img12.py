import cv2
from soln import *
from main import image_decode
square_se=cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
img = cv2.imread('Test Cases/12-mal7-w-felfel.png' ,cv2.IMREAD_GRAYSCALE)
median_filtered = cv2.medianBlur(img, 21)
thresholded_image = Threshold_img(median_filtered)
final_img = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, square_se)
image_decode(final_img)
QR_Print_BA(img , final_img)