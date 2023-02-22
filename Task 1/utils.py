import math

import cv2 as cv
import numpy as np


def return_correct_angles_in_degrees(two_lines):
    """
    :param two_lines: two lines in parametric form
    :return: theta of the two lines in correct format for task1
    """
    if two_lines[0][0] >= 0:
        angle1 = 90 - np.degrees(two_lines[0][1])
    else:
        angle1 = 270 - np.degrees(two_lines[0][1])

    if two_lines[1][0] >= 0:
        angle2 = 90 - np.degrees(two_lines[1][1])
    else:
        angle2 = 270 - np.degrees(two_lines[1][1])

    return angle1, angle2


def show_hough_lines(lines, original_image):
    """
    Given the hough lines and original image will draw the hough lines on the image
    :param lines: the hough lines output by opencv on the original image
    :param original_image: the original image
    :return: None
    """
    for line in lines:
        rho = line[0]
        theta = line[1]

        a = math.cos(theta)
        b = math.sin(theta)

        x0 = a * rho
        y0 = b * rho

        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

        cv.line(original_image, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)
