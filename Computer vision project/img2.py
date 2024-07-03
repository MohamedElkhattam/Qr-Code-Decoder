import cv2
import numpy as np
import matplotlib.pyplot as plt
from soln import *

img = cv2.imread('Test Cases/02-Matsawar-3edel-ya3am.png',cv2.IMREAD_GRAYSCALE)

angle = -8.3
center = (img.shape[1] // 2, img.shape[0] // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]) ,flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
thresholded_image = Threshold_img(rotated_image)

shift_horizontal = 120
translation_matrix = np.float32([[1, 0, shift_horizontal], [0, 1, 15]])
shifted_image = cv2.warpAffine(thresholded_image, translation_matrix, (thresholded_image.shape[1], img.shape[0]),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#Fixing  some black area
height, width = img.shape
crow, ccol = height // 2, width // 2
mask = np.ones((height, width), np.uint8)
mask[crow-506:crow-495, ccol-506:ccol+510] = 0 #Full_Top
mask[crow-506:crow+510, ccol+480:ccol+510] = 0 #Full_Right
mask[crow-506:crow+510, ccol-506:ccol-450] = 0 #Full_Left
mask[crow-506:crow-150, ccol-506:ccol-150] = 0 # Top Light Box 
mask[crow+132:crow+506, ccol-506:ccol-150] = 0 #Bottom Left Box
inverted_mask = 1 - mask
final_img1 = shifted_image.copy()
final_img1[inverted_mask == 1] = 255 #boolean indexing

output_img = np.full_like(final_img1, 255)
num_labels, labels = cv2.connectedComponents(~final_img1)
for label in range(1, num_labels):
    component = np.zeros(final_img1.shape)
    component[labels == label] = 255
    x, y, w, h = cv2.boundingRect(component.astype(np.uint8))
    if x > 600 and y < 100:
        # Extract the black border of the component
        Finder_pattern = final_img1[y:y+h, x:x+w]
        # Place the border at its original position in the output image
        output_img[y:y+h, x:x+w] = Finder_pattern

mirrored_img1 = cv2.flip(output_img,1) #At top
mirrored_img2 = cv2.flip(mirrored_img1,0) # At bottom
mirrored_img = cv2.bitwise_and(mirrored_img1 , mirrored_img2)
translation_matrix1 = np.float32([[1, 0, -20], [0, 1, 0]])
shifted_image1 = cv2.warpAffine(mirrored_img, translation_matrix1, (mirrored_img.shape[1], img.shape[0]))

final_img11 = final_img1.copy()
final_img11[shifted_image1 == 0] = 0

mask2 = np.ones((height, width), np.uint8)
mask2[crow+490:crow+510, ccol-506:ccol+506] = 0 #Full_Bottom
mask2[crow-506:crow-500, ccol-506:ccol+510] = 0 #Full_Top
mask2[crow-506:crow+510, ccol+485:ccol+510] = 0 #Full_Right
mask2[crow-506:crow+510, ccol-506:ccol-492] = 0 #Full_Left

inverted_mask2 = 1 - mask2
final_img = final_img11.copy()
final_img[inverted_mask2 == 1] = 255 

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(final_img, cmap='gray')
plt.show()