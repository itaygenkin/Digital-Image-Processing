import cv2
import sys
import numpy as np


def threshold_image(src_img, threshold):
    for i in range(len(src_img)):
        for j in range(len(src_img[i])):
            if src_img[i, j] > threshold:
                src_img[i, j] = 255
            else:
                src_img[i, j] = 0


def pad_image(src_img, pad_length):
    """
    padding an img
    :param src_img: the image that should be padded
    :param pad_length: the length of the padding (for each side)
    :return: new padded image
    :rtype: cv2.image (np.ndarray, 2d)
    """
    col, row = src_img.shape
    padded_img = np.ones(shape=(col + 2 * pad_length, row + 2 * pad_length), dtype=src_img.dtype) * 255
    for i in range(len(src_img)):
        for j in range(len(src_img[0])):
            padded_img[i + pad_length, j + pad_length] = src_img[i, j]
    return padded_img


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


def erosion(src_img, structure_element, pad_length):
    dest_img = np.ones(shape=src_img.shape, dtype=src_img.dtype) * 255
    for i in range(pad_length, len(dest_img) - pad_length):
        for j in range(pad_length, len(dest_img[i]) - pad_length):
            is_black_pixel = True
            a, b = -1 * pad_length, -1 * pad_length
            while a < pad_length and is_black_pixel:
                while b < pad_length and is_black_pixel:
                    if structure_element[a + pad_length, b + pad_length] > 0 and src_img[i + a, j + b] != 0:
                        is_black_pixel = False
                    b += 1
                a += 1
            if is_black_pixel:
                dest_img[i, j] = 0
    return dest_img


def dilation(src_img, structure_element, pad_length):
    dest_img = np.ones(shape=src_img.shape, dtype=src_img.dtype) * 255

    for i in range(pad_length, len(dest_img) - pad_length):
        for j in range(pad_length, len(dest_img[i]) - pad_length):
            is_black_pixel = False
            a, b = -1 * pad_length, -1 * pad_length

            while a < pad_length and not is_black_pixel:
                while b < pad_length and not is_black_pixel:
                    if structure_element[a + pad_length, b + pad_length] > 0 and src_img[i + a, j + b] == 0:
                        is_black_pixel = True
                    b += 1
                a += 1

            if is_black_pixel:
                dest_img[i, j] = 0
    return dest_img


def negative(src_img):
    """
    negates the color of each pixel in an image
    :param src_img: the image which we want to negate
    :type src_img: 2-dim numpy array
    """
    for i in range(len(src_img)):
        for j in range(len(src_img[0])):
            src_img[i, j] = 255 - src_img[i][j]


def find_max_width(contours_list):
    """
    find the maximum width of each contour
    :param contours_list: list of contours
    :return: list of maximum width of each contour
    """
    width_list = []
    for contour in contours_list:
        min_x = contour[0, 0, 0]
        max_x = contour[0, 0, 0]
        for i in range(len(contour)):
            if min_x > contour[i, 0, 0]:
                min_x = contour[i, 0, 0]
            if max_x < contour[i, 0, 0]:
                max_x = contour[i, 0, 0]
        width_list.append(max_x - min_x)
    return width_list


def find_max_height(contours_list):
    """
    find the maximum height of each contour
    :param contours_list: list of contours
    :return: list of maximum height of each contour
    """
    height_list = []
    for contour in contours_list:
        min_y = contour[0, 0, 1]
        max_y = contour[0, 0, 1]
        for i in range(len(contour)):
            if min_y > contour[i, 0, 1]:
                min_y = contour[i, 0, 1]
            if max_y < contour[i, 0, 1]:
                max_y = contour[i, 0, 1]
        height_list.append(max_y - min_y)
    return height_list


#### START PROCESS ####
# loading the image
img = cv2.imread(sys.argv[1], 0)
original_img = cv2.imread(sys.argv[1], 0)

threshold_image(img, 150)
negative(img)

contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

max_heights = find_max_height(contours)
max_widths = find_max_width(contours)
max_height = max(max_heights)
max_width = max(max_widths)

# brush the demarcation
max_min_max = max(min(max_heights), min(max_widths)) + 1
for i in range(len(contours)):
    if max_heights[i] < max_height * 0.3 and max_widths[i] < max_width * 0.5:
        cv2.drawContours(img, contours, i, 0, max_min_max)

negative(img)

padded_img = pad_image(img, max_min_max // 2)
# create the structure element
se = np.ones(shape=(max_min_max, max_min_max))
for i in range(max_min_max):
    for j in range(max_min_max):
        if i + j < ((max_min_max - 1) // 2) - 1:
            se[i, j] = 0
            se[i, -j - 1] = 0
        elif i + j >= ((3 * max_min_max - 1) // 2) - 1:
            se[i, j] = 0
            se[i, max_min_max - 1 - j] = 0

# opening & pad inverse
res_img = erosion(padded_img, se, max_min_max // 2)
res_img = dilation(res_img, se, max_min_max // 2)
res_img = pad_inverse(res_img, max_min_max // 2)

# printing the original image and the result image
cv2.imshow('original img', original_img)
cv2.imshow('result img', res_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
