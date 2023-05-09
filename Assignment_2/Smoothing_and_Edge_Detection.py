import math
import cv2
import sys
import numpy as np


def normalize_matrix(matrix):
    summ = 0
    for r in matrix:
        summ += sum(r)
    return matrix / summ


def pad_image(src_img, pad_length):
    """
    padding an img with mirroring
    :param src_img: the image that should be padded
    :param pad_length: the length of the padding (for each side)
    :return: new padded image
    :rtype: cv2.image (np.ndarray, 2d)
    """
    col, row = src_img.shape[0], src_img.shape[1]
    new_img = np.zeros(shape=(col + 2 * pad_length, row + 2 * pad_length), dtype=src_img.dtype)

    for i in range(len(new_img)):
        delta_i = i - (len(new_img) - pad_length)

        for j in range(len(new_img[i])):
            delta_j = j - (len(new_img[i]) - pad_length)
            a, b = i - pad_length, j - pad_length

            # upper left corner
            if i < pad_length and j < pad_length:
                a, b = pad_length - j - 1, pad_length - i - 1
            # upper rectangle
            elif i < pad_length and j < len(new_img[i]) - pad_length:
                a = pad_length - i - 1
            # upper right corner
            elif i < pad_length and j >= len(new_img[i]) - pad_length:
                a, b = pad_length - delta_j - 1, len(new_img[i]) - 2 * pad_length - i - 1
            # left rectangle
            elif i < len(new_img) - pad_length and j < pad_length:
                b = pad_length - j - 1
            # right rectangle
            elif i < len(new_img) - pad_length and j >= len(new_img[i]) - pad_length:
                b = len(new_img[i]) - 2 * pad_length - delta_j - 1
            # bottom left corner
            elif i >= len(new_img) - pad_length and j < pad_length:
                a, b = len(src_img) - j - 1, pad_length - delta_i - 1
            # bottom rectangle
            elif i >= len(new_img) - pad_length and j < len(new_img[i]) - pad_length:
                a = len(new_img) - 2 * pad_length - delta_i - 1
            # bottom right corner
            elif i >= len(new_img) - pad_length and j >= len(new_img[i]) - pad_length:
                a, b = pad_length - delta_j - 1, pad_length - delta_i - 1

            new_img[i][j] = src_img[a][b]

    return new_img


def pad_inverse(src_img, pad_length):
    """
    remove the padding of an image
    :param src_img: the image that should be unpadded
    :param pad_length: the number of pixels that should be removed from each row and columns (in each side)
    :return: new image without the padding
    :rtype: cv2.image (np.ndarray, 2d)
    """
    col, row = src_img.shape[0], src_img.shape[1]
    new_img = np.zeros(shape=(col - 2 * pad_length, row - 2 * pad_length), dtype=src_img.dtype)

    for i in range(len(new_img)):
        for j in range(len(new_img[i])):
            new_img[i][j] = src_img[i + pad_length][j + pad_length]
    return new_img


def convert_row(row):
    splitted_row = row.split(',')
    return [int(x) for x in splitted_row if len(x) > 0]


def smooth(src_img, img_filter, pad_size):
    """
    smooth (with any filter matrix) an image (that might be padded)
    :param src_img: the image that might be filtered
    :param img_filter: the filter we use to smooth the image
    :param pad_size: the size of the padding of src_img
    :return: new smoothed image
    """
    smoothed_img = np.zeros(shape=src_img.shape, dtype=src_img.dtype)
    for i in range(pad_size, len(src_img) - pad_size):
        for j in range(pad_size, len(src_img[i]) - pad_size):
            pixel_value = 0
            for a in range(-1 * pad_size, pad_size + 1):
                for b in range(-1 * pad_size, pad_size + 1):
                    pixel_value += src_img[i + a][j + b] * img_filter[a + pad_size][b + pad_size]
            pixel_value = min(255, pixel_value)
            pixel_value = max(0, pixel_value)
            smoothed_img[i][j] = pixel_value
    return smoothed_img


def build_gaussian_matrix(filter_size):
    matrix = np.zeros(shape=(filter_size, filter_size))
    size = (filter_size - 1) // 2
    my_sum = 0
    for i in range(size * -1, size + 1):
        for j in range(size * -1, size + 1):
            gaussian = (math.exp(-1 * (i*i + j*j) / 2)) / 2 * math.pi
            gaussian = math.ceil(gaussian * 100) / 100
            matrix[i + size][j + size] = gaussian
            my_sum += gaussian
    return (matrix / my_sum).round(decimals=4)


def LoG(filter_size):
    """
    compute the values that should be in the filter matrices of the edge detection
    and place them into 2 matrices
    :param filter_size: the size of the matrices it builds
    :return: build two matrices for the edge detection
    :rtype: np.ndarray (2d)
    """
    kernels_and_values = {3: 2, 5: 20, 7: 780, 9: 132600}
    G_x = np.zeros(shape=(filter_size, filter_size))
    G_y = np.zeros(shape=(filter_size, filter_size))
    size = (filter_size - 1) // 2
    for i in range(size * -1, size + 1):
        for j in range(size * -1, size + 1):
            if i != 0 or j != 0:
                G_x[i + size, j + size] = i / (i*i + j*j)
                G_y[i + size, j + size] = j / (i*i + j*j)
    return G_x * kernels_and_values[filter_size], G_y * kernels_and_values[filter_size]


def compute_sobel(A, B, threshold):
    """
    computing sobel for edge detection filtering.
    :param A: matrix, np.ndarray. Filter matrix in first direction (X).
    :param B: matrix, np.ndarray. Filter matrix in second direction (Y).
    :param threshold: the threshold which determine the pixel value.
    :return: filtered cv2.image by the edge detection method.
    """
    output = np.zeros(shape=A.shape, dtype=A.dtype)
    for i in range(len(A)):
        for j in range(len(A[i])):
            pixel_value = int(math.sqrt(A[i, j] ** 2 + B[i, j] ** 2))
            if pixel_value > threshold:
                output[i, j] = 255
            else:
                output[i, j] = 0
    return output


def different_images(src_img, dest_img):
    """
    compute the difference in the pixels values in two images
    :param src_img: cv2.image (np.ndarray, 2d)
    :param dest_img: cv2.image (np.ndarray, 2d)
    :return: new image with the difference values in each pixel
    :rtype: cv2.image
    """
    dif_img = np.zeros(src_img.shape, dtype=src_img.dtype)
    for i in range(len(src_img)):
        for j in range(len(src_img[i])):
            dif_img[i][j] = max(0, int(src_img[i][j]) - int(dest_img[i][j]))
    return dif_img


#### Image processing ####
img = cv2.imread(sys.argv[1], 0)
# img = cv2.resize(img, (640, 480))
# let the user choose the filter
kind_of_filter = input("Choose filter: \n1 - Smoothing\n2 - Edge Detection\n")
filter_size = int(input("Choose filter size: 3\\5\\7\\9\n"))


# check input validity
while filter_size not in [3, 5, 7, 9]:
    filter_size = int(input("Choose filter size: 3\\5\\7\\9\n"))

# smooth case
if kind_of_filter == '1':
    # build a default filter matrix
    filter_matrix = build_gaussian_matrix(filter_size)
    print('Default filter:\n', filter_matrix)

    set_filter = input('Set filter? (y/n)\n')
    if set_filter == 'y':
        user_matrix = []
        for r in range(filter_size):
            row = input(f'Write row {r} values separated by \',\'')
            user_matrix.append(convert_row(row))
        user_filter = np.array(user_matrix)
        filter_matrix = normalize_matrix(user_filter)
        print('New filter\n', user_filter)

    padded_img = pad_image(img, (filter_size - 1) // 2)
    padded_img = smooth(padded_img, filter_matrix, (filter_size - 1) // 2)
    res_image = pad_inverse(padded_img, (filter_size - 1) // 2)

# edge detection case
else:
    threshold_bound = 128
    print(f'Default threshold: {threshold_bound}\n')
    set_threshold = input('Set threshold? (y/n)\n')
    if set_threshold == 'y':
        threshold_bound = int(input('Threshold: '))

    padded_img = pad_image(img, (filter_size - 1) // 2)
    Gx_filter, Gy_filter = LoG(filter_size)

    Gx = smooth(padded_img, Gx_filter, (filter_size - 1) // 2)
    Gy = smooth(padded_img, Gy_filter, (filter_size - 1) // 2)

    Gx = pad_inverse(Gx, (filter_size - 1) // 2)
    Gy = pad_inverse(Gy, (filter_size - 1) // 2)
    res_image = compute_sobel(Gx, Gy, threshold_bound)

diff_img = different_images(img, res_image)

cv2.imshow('original image', img)
cv2.imshow('result image', res_image)
cv2.imshow('difference image', diff_img)
cv2.waitKey(0)
