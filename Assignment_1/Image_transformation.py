import numpy as np
import cv2
import sys

# Globals
x1, y1, x2, y2, x3, y3 = (-1, -1, -1, -1, -1, -1)
num_of_clicks = 0


def nearest_neighbor(src_img, point):
    nearest_neighbor_x = round(point[0])
    nearest_neighbor_y = round(point[1])
    if nearest_neighbor_x < 0 or nearest_neighbor_x > len(src_img) - 1 or nearest_neighbor_y < 0 or \
            nearest_neighbor_y > len(src_img[0]) - 1:
        return 0
    else:
        return src_img[nearest_neighbor_x][nearest_neighbor_y]


def bilinear_interpolation(src_img, point):
    # find the 4 closest pixels points
    upper_left = int(point[0]), int(point[1])
    upper_right = int(point[0]) + 1, int(point[1])
    bottom_left = int(point[0]), int(point[1]) + 1
    bottom_right = int(point[0]) + 1, int(point[1]) + 1

    # compute the distance in each dimension from the closest upper left pixel
    alpha = point[0] - upper_left[0]
    beta = point[1] - upper_left[1]

    if upper_left[0] < 0 or upper_left[0] > len(src_img) - 1 or upper_left[1] < 0 or upper_left[1] > len(src_img[0]) - 1:
        return 0
    else:
        x = (1 - alpha) * (1 - beta) * upper_left[0] + alpha * (1 - beta) * upper_right[0] + \
            (1 - alpha) * beta * bottom_left[0] + alpha * beta * bottom_right[0]
        y = (1 - alpha) * (1 - beta) * upper_left[1] + alpha * (1 - beta) * upper_right[1] + \
            (1 - alpha) * beta * bottom_left[1] + alpha * beta * bottom_right[1]

    return src_img[round(x)][round(y)]


def cubic_solver(x):
    x = abs(x)
    if 0 <= x <= 1:
        return 1.5 * (x ** 3) - 2.5 * (x ** 2) + 1
    elif 1 < x <= 2:
        return -0.5 * (x ** 3) + 2.5 * (x ** 2) - 4 * x + 2
    else:
        return 0


def cubic_interpolation(src_img, point):
    x, y = point[0], point[1]
    dx, dy = abs(x - round(x)), abs(y - round(y))
    gray_sum = 0
    bgr_sum = [0, 0, 0]

    if int(x) - 1 < 0 or int(x) + 2 > len(src_img) - 1 or int(y) < 0 or int(y) > len(src_img[0]) - 1:
        if len(src_img.shape) == 2:
            return 0
        else:
            return [0, 0, 0]
    else:
        for i in range(-1, 3):
            for j in range(-1, 3):
                CaX = cubic_solver(j + dx)
                CaY = cubic_solver(i + dy)
                if len(src_img.shape) == 2:  # Grayscale
                    gray_sum += src_img[round(x) + j, round(y) + i] * CaX * CaY
                else:  # BGR
                    bgr_sum[0] += src_img[round(x) + j, round(y) + i][0] * CaX * CaY
                    bgr_sum[1] += src_img[round(x) + j, round(y) + i][1] * CaX * CaY
                    bgr_sum[2] += src_img[round(x) + j, round(y) + i][2] * CaX * CaY
    # Grayscale case
    if len(src_img.shape) == 2:
        if gray_sum > 255:
            return 255
        elif gray_sum < 0:
            return 0
        else:
            return gray_sum
    # BGR case
    else:
        if bgr_sum[0] > 255:
            bgr_sum[0] = 255
        elif bgr_sum[0] < 0:
            bgr_sum[0] = 0

        if bgr_sum[1] > 255:
            bgr_sum[1] = 255
        elif bgr_sum[1] < 0:
            bgr_sum[1] = 0

        if bgr_sum[2] > 255:
            bgr_sum[2] = 255
        elif bgr_sum[2] < 0:
            bgr_sum[2] = 0

        return bgr_sum


def parabola(y):
    """
    calculate parabola interpolation at the three y's points the user gave,
    and substitute y
    :param y: a point where the user wants to compute tha value of the parabola
    :return: the value of the parabola at the point y
    :rtype: double
    """
    a = (y - y1) / (y3 - y1)
    b = (y - y2) / (y3 - y2)
    c = (y - y3) / (y1 - y3)
    d = (y - y2) / (y1 - y2)
    e = (y - y3) / (y2 - y3)
    f = (y - y1) / (y2 - y1)
    x_m = (x1 + x2) // 2
    return (a * b) * x3 + (c * d) * x_m + (e * f) * x_m


def inverse_transformation(src_img):
    """
    create 3 images (one for each method) that takes the pixels from src_img
    using inverse transformation
    :param src_img: an image which the pixels are taken for the new images
    :type src_img: numpy ndarray (2 or 3 dimensions)
    """
    # find the width and height of src_img
    width, height = src_img.shape[:2]

    # initialize 3 matrices (images) for the 3 interpolations
    nn_image = np.zeros((width, height, 3), src_img.dtype)
    bilinear_image = np.zeros((width, height, 3), src_img.dtype)
    cubic_image = np.zeros((width, height, 3), src_img.dtype)

    rect_half_width = (x1 - x2) / 2
    rect_half_x = round((x1 + x2) / 2)

    for j in range(height):
        for i in range(width):
            parab_x = parabola(i)
            relative_left_parab_x = parab_x - x2
            relative_right_parab_x = parab_x - rect_half_x
            # if the pixel is in rect range
            if x2 <= j <= x1 and y1 <= i <= y2:
                # x vals up to parabola line
                if j <= parab_x:
                    # compute (y,x) pixel from original image
                    x = (((j - x2) / relative_left_parab_x) * rect_half_width) + x2
                    y = i

                    # Interpolation in each method:
                    nn_image[i][j] = nearest_neighbor(src_img, (y, x))
                    bilinear_image[i][j] = bilinear_interpolation(src_img, (y, x))
                    cubic_image[i][j] = cubic_interpolation(src_img, (y, x))
                # x vals from rect medial line
                else:
                    x = ((1 - ((x1 - j) / (rect_half_width - relative_right_parab_x)))
                         * rect_half_width) + rect_half_x
                    y = i
                    # Interpolations in each method:
                    nn_image[i][j] = nearest_neighbor(src_img, (y, x))
                    bilinear_image[i][j] = bilinear_interpolation(src_img, (y, x))
                    cubic_image[i][j] = cubic_interpolation(src_img, (y, x))

            # if the pixel is not in rect range take the value from original image as it is
            else:

                nn_image[i][j] = src_img[i][j]
                bilinear_image[i][j] = src_img[i][j]
                cubic_image[i][j] = src_img[i][j]
    # show images
    cv2.imshow("nearest neighbour", nn_image)
    cv2.imshow("bilinear", bilinear_image)
    cv2.imshow("cubic", cubic_image)
    cv2.waitKey(0)


def set_points():
    """
    swap user's points so that there's only one format
    """
    global x1, x2, y1, y2
    if x2 > x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1


def click_event(event, x, y, flags, params):
    global x1, x2, y1, y2, x3, y3, num_of_clicks, img

    # checking for left mouse clicking
    if event == cv2.EVENT_LBUTTONDOWN:
        # first time clicking
        if num_of_clicks == 0:
            x1, y1 = x, y
            num_of_clicks += 1

        # second time clicking
        elif num_of_clicks == 1:
            x2, y2 = x, y
            set_points()
            num_of_clicks += 1
            rec = cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2),
                                color=(0, 0, 0), thickness=2)
            line = cv2.line(img, pt1=((x1 + x2) // 2, y1),
                            pt2=((x1 + x2) // 2, y2), color=(0, 0, 0), thickness=2)
            cv2.imshow('image', img)

        # third click to create an ellipse
        elif num_of_clicks == 2:
            x3, y3 = x, y
            num_of_clicks += 1
            ellipse_node = abs(x3 - ((x1 + x2) // 2))
            axes_length = (ellipse_node, (y2 - y1) // 2)
            angle = 0
            if x3 < ((x1 + x2) // 2):
                angle = 180
            ellipse = cv2.ellipse(img, center=[(x1 + x2) // 2, (y1 + y2) // 2], axes=axes_length, angle=angle,
                                  startAngle=90, endAngle=-90, color=(0, 0, 0), thickness=2)
            cv2.imshow('image', img)

        # forth click to call inverse_transformation
        elif num_of_clicks == 3:
            num_of_clicks += 1
            inverse_transformation(clean_img)

    # checking for a right mouse clicking (and close all windows)
    if event == cv2.EVENT_RBUTTONDOWN:
        if num_of_clicks >= 4:
            cv2.destroyAllWindows()
        else:
            num_of_clicks = 0
            img = cv2.imread(sys.argv[1])


img = cv2.imread(sys.argv[1])
img = cv2.resize(img, (640, 480))
clean_img = cv2.imread(sys.argv[1])
clean_img = cv2.resize(img, (640, 480))
cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
