import cv2
import numpy as np
import matplotlib.pyplot as plt
from soln import *

imgs_before=[]
imgs_before.append(cv2.imread("Test cases/01-Getting-started.png" ,cv2.IMREAD_GRAYSCALE))
imgs_before.append(cv2.imread("Test cases/03-Leffy-bina-ya-donya.png", cv2.IMREAD_GRAYSCALE))
imgs_before.append(cv2.imread("Test cases/04-Black-mirror.png", cv2.IMREAD_GRAYSCALE))
imgs_before.append(cv2.imread("Test cases/09-My-phone-fell-while-taking-this-one-...-or-did-it.png" ,cv2.IMREAD_GRAYSCALE))
imgs_before.append(cv2.imread("Test Cases/11-weewooweewooweewoo.png", cv2.IMREAD_GRAYSCALE))
imgs_before.append(cv2.imread("Test Cases/08-Compresso-Espresso.png",cv2.IMREAD_GRAYSCALE))
imgs_before.append(cv2.imread("Test Cases/10-Gone-With-The-Wind.png",cv2.IMREAD_GRAYSCALE))
# imgs_before.append(cv2.imread("Test Cases/06-Railfence-cipher.png",cv2.IMREAD_GRAYSCALE))
# imgs_before.append(cv2.imread("Test Cases/14-BANANAAA!!!.png",cv2.IMREAD_GRAYSCALE))

Default_Version1_Template= cv2.imread("Test cases/01-Getting-started.png" ,cv2.IMREAD_GRAYSCALE)
Default_Finder_Pattern_bottom = Component_Extraction(Default_Version1_Template)[0]
finders2_detector = cv2.bitwise_and(Default_Finder_Pattern_bottom,flip_image(Default_Finder_Pattern_bottom,1))
Default_Finder_Pattern_Top = Component_Extraction(Default_Version1_Template , 600 , 100)[0]
flag = True

def image_decode(final_img):
    start_row = -1
    start_col = -1
    end_row = -1
    end_col = -1

    for row_index, row in enumerate(final_img):
        for pixel in row:
            if pixel != 255:
                start_row = row_index
                break
        if start_row != -1:
            break

    for row_index, row in enumerate(final_img[::-1]): # reverse rows --> first go last
        for pixel in row:
            if pixel != 255:
                end_row = final_img.shape[0] - row_index
                break
        if end_row != -1:
            break

    for col_index, col in enumerate(cv2.transpose(final_img)):
        for pixel in col:
            if pixel != 255:
                start_col = col_index
                break
        if start_col != -1:
            break

    for col_index, col in enumerate(cv2.transpose(final_img)[::-1]):
        for pixel in col:
            if pixel != 255:
                end_col = final_img.shape[1] - col_index
                break
        if end_col != -1:
            break
    print(start_row, end_row, start_col, end_col)
    
    qr_no_quiet_zone = final_img[start_row:end_row, start_col:end_col]
    fig = plt.figure(figsize=(5, 5))
    plt.xticks([], [])
    plt.yticks([], [])
    fig.get_axes()[0].spines[:].set_color('red')
    fig.get_axes()[0].spines[:].set_linewidth(40)
    fig.get_axes()[0].spines[:].set_position(("outward", 20))
    plt.imshow(qr_no_quiet_zone, cmap='gray')

    size = 0
    for pixel in qr_no_quiet_zone[0]:
        if (pixel != 0): break
        size += 1

    grid_cell_size = round(size / 7)
    if grid_cell_size == 0:
        grid_cell_size = 44  
    print(grid_cell_size)

    grid_cells_num = round(qr_no_quiet_zone.shape[0]/grid_cell_size)
    # grid_cells_num = 21
    print(grid_cells_num)
    
    # Reshape the qr_no_quiet_zone into grid cells
    qr_cells = qr_no_quiet_zone[:grid_cells_num * grid_cell_size, 
        :grid_cells_num * grid_cell_size].reshape(
        (grid_cells_num, grid_cell_size, grid_cells_num, grid_cell_size)).swapaxes(1, 2)

    # Visualize the grid cells
    print(grid_cells_num , grid_cells_num)
    _, axes = plt.subplots(grid_cells_num, grid_cells_num, figsize=(5, 5))
    for i, row in enumerate(axes):
        for j, col in enumerate(row):
            col.imshow(qr_cells[i][j], cmap="gray", vmin=0, vmax=255)
            col.get_xaxis().set_visible(False)
            col.get_yaxis().set_visible(False)
            col.spines[:].set_color('red')
    qr_cells_numeric = np.ndarray((grid_cells_num, grid_cells_num), dtype=np.uint8)
    for i, row in enumerate(qr_cells):
        for j, cell in enumerate(row):
            qr_cells_numeric[i, j] = (np.median(cell) // 255)
    print(qr_cells_numeric)

    # We want row #8
    qr_cells_numeric[8]
    # The first two bits determine the error correction level
    # Level L (Low)         [11]	7%  of data bytes can be restored.
    # Level M (Medium)      [10]	15% of data bytes can be restored.
    # Level Q (Quartile)    [01]	25% of data bytes can be restored.
    # Level H (High)        [00]	30% of data bytes can be restored.
    ecl = [int(not(c)) for c in qr_cells_numeric[8, 0:2]]
    # Why "not"? Because the standard uses '1's for black and '0's for white
    #
    # "A dark module is a binary one and a light module is a binary zero."
    #  - ISO/IEC 18004:2000(E)
    #
    # In image processing, we use them the other way.. Hence the inversion
    print(ecl)
    # Dictionary of all masks and their equivalent formulae
    MASKS = {
        "000": lambda i, j: (i * j) % 2 + (i * j) % 3 == 0,
        "001": lambda i, j: (i / 2 + j / 3) % 2 == 0,
        "010": lambda i, j: ((i * j) % 3 + i + j) % 2 == 0,
        "011": lambda i, j: ((i * j) % 3 + i * j) % 2 == 0,
        "100": lambda i, j: i % 2 == 0,
        "101": lambda i, j: (i + j) % 2 == 0,
        "110": lambda i, j: (i + j) % 3 == 0,
        "111": lambda i, j: j % 3 == 0,
    }

    # Same row as above, the three cells after the ecl cells (converted to a string)
    mask = [int(not(c)) for c in qr_cells_numeric[8, 2:5]]
    mask_str = ''.join([str(c) for c in mask])
    print(mask_str)
    # Same row as above, but we want cells #5 and #7 (#6 is always set to 0),
    #  followed by column #8 from cell #0 in it to cell #7 (and skipping #6)
    fec = []
    fec.append(qr_cells_numeric[8, 5])
    fec.append(qr_cells_numeric[8, 7])
    fec.extend(qr_cells_numeric[0:6, 8])
    fec.extend(qr_cells_numeric[7:9, 8])
    fec = [int(not(c)) for c in fec]
    print(fec)
    # So in total we have the following 15 bits of format info from our QR code
    print(ecl, mask, fec)
    # Let's cross-check with our example
    _, axes = plt.subplots(grid_cells_num, grid_cells_num, figsize=(5, 5))
    for i, row in enumerate(axes):
        for j, col in enumerate(row):

            col.get_xaxis().set_visible(False)
            col.get_yaxis().set_visible(False)
            if (i == 8 and j <= 8) or (i <= 8 and j == 8):
                if (i != 6) and (j != 6):
                    col.imshow(qr_cells[i][j], cmap="gray", vmin=0, vmax=255)
                    col.spines[:].set_color('red')
                    continue
            col.imshow(qr_cells[i][j], cmap="gray", vmin=-1275, vmax=510)
    # However..... You need to XOR that with the "format mask": 101010000010010
    ecl[0] ^= 1
    mask[0] ^= 1
    mask[2] ^= 1
    fec[5] ^= 1
    fec[8] ^= 1

    # And now we print...
    print(ecl, mask, fec)
    # Before we proceed, let's write a function for masking to make our lives easier
    UP, UP_ENC, DOWN, CW, CCW = range(5)  # A rather old-fashioned pythonic "Enum"

    def apply_mask(data_start_i, data_start_j, direction):
        '''
        data_start_i/j represent the first cell's coords in its respective direction
        direction is the masking direction, up(-enc)/down/clockwise/anti-clockwise
        '''
        result = []
        row_offsets = []
        col_offsets = []
        if (direction in [UP, UP_ENC]):
            row_offsets = [0,  0, -1, -1, -2, -2, -3, -3]
            col_offsets = [0, -1,  0, -1,  0, -1,  0, -1]
        if (direction == DOWN):
            row_offsets = [0,  0,  1,  1,  2,  2,  3,  3]
            col_offsets = [0, -1,  0, -1,  0, -1,  0, -1]
        if (direction == CW):
            row_offsets = [0,  0,  1,  1,  1,  1,  0,  0]
            col_offsets = [0, -1,  0, -1, -2, -3, -2, -3]
        if (direction == CCW):
            row_offsets = [0,  0, -1, -1, -1, -1,  0,  0]
            col_offsets = [0, -1,  0, -1, -2, -3, -2, -3]

        for i, j in zip(row_offsets, col_offsets):
            cell = qr_cells_numeric[data_start_i+i, data_start_j+j]
            result.append(int(cell if MASKS[mask_str](data_start_i+i, data_start_j+j) else not cell))

        return result[:4] if direction == UP_ENC else result
    enc = apply_mask(grid_cells_num-1, grid_cells_num-1, UP_ENC)
    print(enc)
    len = apply_mask(grid_cells_num-3, grid_cells_num-1, UP)
    print(len)
    data_starting_indices = [
        [grid_cells_num-7, grid_cells_num-1, UP],
        [grid_cells_num-11, grid_cells_num-1, CCW],
        [grid_cells_num-10, grid_cells_num-3, DOWN],
        [grid_cells_num-6, grid_cells_num-3, DOWN],
        [grid_cells_num-2, grid_cells_num-3, CW],
        [grid_cells_num-3, grid_cells_num-5, UP],
        [grid_cells_num-7, grid_cells_num-5, UP],
        [grid_cells_num-11, grid_cells_num-5, CCW],
        [grid_cells_num-10, grid_cells_num-7, DOWN],
        [grid_cells_num-6, grid_cells_num-7, DOWN],
        [grid_cells_num-2, grid_cells_num-7, CW],
        [grid_cells_num-3, grid_cells_num-9, UP],
        [grid_cells_num-7, grid_cells_num-9, UP],
        [grid_cells_num-11, grid_cells_num-9, UP],
        [grid_cells_num-16, grid_cells_num-9, UP],
        [grid_cells_num-20, grid_cells_num-9, CCW],
        [grid_cells_num-19, grid_cells_num-11, DOWN],
        [grid_cells_num-14, grid_cells_num-11, DOWN],
        [grid_cells_num-10, grid_cells_num-11, DOWN],
        [grid_cells_num-6, grid_cells_num-11, DOWN],
        # Hmm..? I actually don't know how to proceed now lol
    ]
    ans = ''
    for a, b, d in data_starting_indices:
        bits = apply_mask(a, b, d)
        bit_string = ''.join([str(bit) for bit in bits])
        if bit_string[:4] == "0000":
            print(f'{bit_string[:4]} = 0 (NULL TERMINATOR)')
            break
        ans += chr(int(bit_string, 2)) # converts to binary to int, then to ASCII
        print(f'{bit_string} = {ans[-1]}')

    print(f'\nDecoded string: {ans}')

plt.figure(figsize=(10, 10))
for i , img in enumerate(imgs_before):
    plt.subplot(2 , 4 , i+1)
    plt.imshow(img ,cmap='gray')
plt.show()

plt.figure(figsize=(10, 5))
for i , img in enumerate(imgs_before):
# Check for need of Vertical flip of image / Inversion
    bottom_pattern = Component_Extraction(img)[0]
    if bottom_pattern.all() != Default_Finder_Pattern_bottom.all() and IS_Inverted(img):
        img = flip_image(~img , 1)
# Check for Problem in frequency domain
    if IS_Low_Frequency(img):
        img = pattern_fix_inFreqDomain(img)[0]
        img = Threshold_img(img)
        flag = False
# Check for Blurred image
    if not IS_Blurred(img , 1) and IS_Blurred(img , 50):
        img = sharpening_filter(img)
        flag = False
# Check for need Rotation in image
    try:
        Double_finder_at_bottom = Component_Extraction(img, 900, 600)[0]
        TopL_Check = Component_Extraction(img, 600, 100)[0]
        differnce_between_images = finders2_detector - Double_finder_at_bottom
        if np.all(differnce_between_images == 0):  
            if TopL_Check.all() == Default_Finder_Pattern_Top.all():
                Vertical_Flipped = flip_image(img, 1)
                img = flip_image(Vertical_Flipped, 0)
                flag = False
    except TypeError:
        pass
    img = Threshold_img(img)
#Final One Finally
    try:
        if flag:
            img = Open_Close_img_21(img)
            img = Component_Extraction(img)[1]            
    except TypeError :
        pass
    flag = True
    image_decode(img)