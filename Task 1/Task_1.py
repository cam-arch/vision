import os

import cv2 as cv
import natsort
import numpy as np

from utils import return_correct_angles_in_degrees, show_hough_lines


def remove_parallel_lines(hough_lines):
    """
    remove any parallel lines (those with same theta)
    :param hough_lines: the hough lines output by opencv on an image
    :return: hough lines without any parallel lines
    """
    if hough_lines is None:
        return None

    hough_lines = hough_lines.reshape(-1, hough_lines.shape[-1])
    unique_thetas, indices = np.unique(hough_lines[:, 1], return_index=True)
    return hough_lines[indices]


def run_task_1(path_to_images):
    """
    Performs task 1 in the spec (returns the angle between two lines on an image)
    :param path_to_images: where the images are stored
    :return: None
    """
    for image in natsort.natsorted(os.listdir(path_to_images), reverse=False):
        image_ = cv.imread('./angle/' + image, cv.IMREAD_GRAYSCALE)
        edges = cv.Canny(image_, 50, 200, None, 3)
        edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

        potential_thresholds = np.linspace(80, 200, 100, dtype=int)
        lines = None
        for threshold in potential_thresholds:
            lines = cv.HoughLines(edges, 1, np.pi / 180, threshold, None, 0, 0)
            lines = remove_parallel_lines(lines)
            if lines is not None and len(lines) == 2:
                break

        if lines is not None and len(lines) == 2:
            show_hough_lines(lines, edges_bgr)

            print(image)
            cv.imshow('Image with line detection', edges_bgr)
            cv.waitKey(0)

            angle1, angle2 = return_correct_angles_in_degrees(lines)
            print("Line 1: " + str(angle1))
            print("Line 2: " + str(angle2))
            print("Angle between them: " + str(round(abs(angle1 - angle2))))

        cv.destroyAllWindows()


if __name__ == '__main__':
    # UPDATE PATH AS REQUIRED
    path = "./angle"
    run_task_1(path)
