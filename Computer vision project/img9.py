import cv2
import matplotlib.pyplot as plt
from soln import *

img = cv2.imread("Test cases/09-My-phone-fell-while-taking-this-one-...-or-did-it.png" ,cv2.IMREAD_GRAYSCALE)
sharpened_image =sharpening_filter(img)
final_img =Threshold_img(sharpened_image)
QR_Print_BA(img , final_img)
plt.hist(img.ravel(),256,[0,256])
