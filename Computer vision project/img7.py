import cv2
import numpy as np
import matplotlib.pyplot as plt
from soln import *

img = cv2.imread('Test Cases/07-THE-MIGHTY-FINGER.png',cv2.IMREAD_GRAYSCALE)

threshold_img = Threshold_img(img)
#Fixing  some black area
height, width = img.shape
crow, ccol = height // 2, width // 2
mask = np.ones((height, width), np.uint8)
mask[crow-506:crow-50, ccol-506:ccol-130] = 0 #Full_Right
inverted_mask = 1 - mask
final_img1 = threshold_img.copy()
final_img1[inverted_mask == 1] = 255 #boolean indexing

#Begining connected component Extraction
output_img = np.full_like(threshold_img, 255)
num_labels, labels = cv2.connectedComponents(~threshold_img)
for label in range(1, num_labels):
    component = np.zeros(threshold_img.shape)
    component[labels == label] = 255
    x, y, w, h = cv2.boundingRect(component.astype(np.uint8))
    if x < 100  and y >600:
        Finder_pattern = threshold_img[y:y+h, x:x+w]
        output_img[y:y+h, x:x+w] = Finder_pattern

mirrored_img = cv2.flip(output_img,0) # 1 for vertical , 0 for horizontal
final_img = final_img1.copy()
final_img[mirrored_img == 0] = 0

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(final_img, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(inverted_mask, cmap='gray')
plt.show()
