import cv2
import numpy as np
import matplotlib.pyplot as plt

square_se=cv2.getStructuringElement(cv2.MORPH_RECT, (21,21))
def Component_Extraction(final_img1 , x_coor = 100 , y_coor = 600):
    try:
    #Begining connected component Extraction
        output_img = np.full_like(final_img1, 255)
        num_labels, labels = cv2.connectedComponents(~final_img1)
    except TypeError:
        return None
    for label in range(0, num_labels):
        component = np.zeros(final_img1.shape)
        component[labels == label] = 255
        x, y, w, h = cv2.boundingRect(component.astype(np.uint8))
        # Finding Finder pattern based on it's coordinates
        # (Top Right--> X > 600 , Y < 100) , (Bottom left X < 100 , Y > 600)
        if x_coor == 600 and y_coor == 100:
            if x > x_coor and y < y_coor :
            # Extract the black border of the component
                Finder_pattern = final_img1[y:y+h, x:x+w]
            # Place the border at its original position in the output image
                output_img[y:y+h, x:x+w] = Finder_pattern
        else:
            if x < x_coor and y > y_coor :
                # Extract the black border of the component
                Finder_pattern = final_img1[y:y+h, x:x+w]
                # Place the border at its original position in the output image
                output_img[y:y+h, x:x+w] = Finder_pattern

    mirrored_img1 = cv2.flip(output_img,0) # 1 for vertical , 0 for horizontal
    mirrored_img2 = cv2.flip(mirrored_img1,1) # 1 for vertical , 0 for horizontal
    mirrored_img = cv2.bitwise_and(mirrored_img1 , mirrored_img2)
    # Apply the mask to the original image to show only the black box
    final_img = final_img1.copy()
    final_img[mirrored_img == 0] = 0
    return  output_img , final_img 

def Threshold_img (img):
    mean_intensity = np.mean(img)
    std_dev = np.std(img)
    threshold_value = mean_intensity + (std_dev * 0.01)
    thresholded_image = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)[1]
    return thresholded_image

def Open_Close_img_21(img):
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, square_se)
    final_img1 = cv2.morphologyEx(open, cv2.MORPH_CLOSE, square_se)
    return final_img1

def Close_Open_img_21(img):
    Close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, square_se)
    final_img1 = cv2.morphologyEx(Close, cv2.MORPH_OPEN, square_se)
    return final_img1

def perspective_transform_and_warp(matrix1, img):
    matrix2 = np.float32([[0,0] ,[924,0] , [0,924] , [924,924]])
    new_matrix = cv2.getPerspectiveTransform(matrix1, matrix2)
    warped_img = cv2.warpPerspective(img, new_matrix, (924, 924))  # Use original image size
    return warped_img

def flip_image(img , num):
    return cv2.flip(img , num) #1 for vertical and  0 for horizontal

def pattern_fix_inFreqDomain(img):
    ft_img = np.fft.fft2(img)
    centered_ft_img = np.fft.fftshift(ft_img) #(-1)^x+y
    magnitude_spectrum = np.log(np.abs(centered_ft_img) + 1)
    
    height, width = img.shape
    crow, ccol = height // 2, width // 2
    mask = np.ones((height, width), np.uint8)
    mask[crow-2:crow+2, ccol+10:ccol+16] = 0
    mask[crow-2:crow+2, ccol-16:ccol-9] = 0 

    f_shift_masked = centered_ft_img * mask
    Unshifted_image = np.fft.ifftshift(f_shift_masked)
    image_restored = np.fft.ifft2(Unshifted_image).real
    visualization_img =  np.abs(image_restored)
    return visualization_img , magnitude_spectrum

def sharpening_filter(img):
    sharpening_Filter = np.array([[0, -2, 0],[-2, 10, -2],[0, -2, 0]])
    return cv2.filter2D(img, -1, sharpening_Filter)

def IS_Blurred(img , blur_threshold):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = np.var(laplacian)
    return variance < blur_threshold

def IS_Inverted(img):
    total_pixels = img.size
    black_pixels = np.count_nonzero(img == 0)  
    black_pixel_percentage = (black_pixels / total_pixels) * 100
    return  black_pixel_percentage > 50

def IS_Low_Frequency(img):
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    frequency= np.sum(magnitude_spectrum) / (img.shape[0] * img.shape[1])
    return 3400<frequency<4000

def QR_Print_BA(img , final_img):
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(final_img, cmap='gray')
    plt.show()