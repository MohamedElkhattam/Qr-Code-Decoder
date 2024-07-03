import cv2
import matplotlib.pyplot as plt
from soln import *

img = cv2.imread("Test cases/11-weewooweewooweewoo.png", cv2.IMREAD_GRAYSCALE)
Visualization_img = pattern_fix_inFreqDomain(img)[0]
final_img = Threshold_img(Visualization_img)
QR_Print_BA(img , final_img)
plt.hist(img.ravel(),256,[0,256])