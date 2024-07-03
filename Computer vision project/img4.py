import cv2
from soln import *

img = cv2.imread('Test cases/04-Black-mirror.png', cv2.IMREAD_GRAYSCALE)
vertical_fliped_img = flip_image(~img , 1)
final_img = Threshold_img(vertical_fliped_img)
QR_Print_BA(img , final_img)