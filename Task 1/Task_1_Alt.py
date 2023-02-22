import math
import os

import cv2 as cv
import natsort
import numpy as np

from utils import return_correct_angles_in_degrees


def find_two_lines(lines):
    maxDiff = 0
    maxDiffPos = 0
    cumRho1 = 0
    cumRho2 = 0
    cumTheta1 = 0
    cumTheta2 = 0

    sortedLines = sorted(lines, key=lambda x: x[0][0])

    for i in range(len(sortedLines) - 1):
        if (sortedLines[i + 1][0][0] - sortedLines[i][0][0]) > maxDiff:
            maxDiffPos = i + 1
            maxDiff = sortedLines[i + 1][0][0] - sortedLines[i][0][0]

    for j in range(0, maxDiffPos):
        cumRho1 += sortedLines[j][0][0]
        cumTheta1 += sortedLines[j][0][1]

    cumRho1 /= maxDiffPos
    cumTheta1 /= maxDiffPos

    for k in range(maxDiffPos, len(sortedLines)):
        cumRho2 += sortedLines[k][0][0]
        cumTheta2 += sortedLines[k][0][1]

    cumRho2 /= len(sortedLines) - maxDiffPos
    cumTheta2 /= len(sortedLines) - maxDiffPos

    return [[cumRho1, cumTheta1], [cumRho2, cumTheta2]]


for image in natsort.natsorted(os.listdir('./angle'), reverse=False):
    image_ = cv.imread('./angle/' + image, cv.IMREAD_GRAYSCALE)
    (thresh, blackAndWhiteImage) = cv.threshold(image_, 127, 255, cv.THRESH_BINARY)
    imageUnderlay = cv.cvtColor(image_, cv.COLOR_GRAY2BGR)
    lines = cv.HoughLines(blackAndWhiteImage, 1.1, np.pi / 150, 120, None, 0, 0)

    if lines is not None:
        twoLines = find_two_lines(lines)

        for line in twoLines:
            rho = line[0]
            theta = line[1]

            a = math.cos(theta)
            b = math.sin(theta)

            x0 = a * rho
            y0 = b * rho

            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

            cv.line(imageUnderlay, pt1, pt2, (0, 255, 0), 1, cv.LINE_AA)

        print(image)
        cv.imshow('Image with Line Detection', imageUnderlay)
        cv.waitKey(0)

        angle1, angle2 = return_correct_angles_in_degrees(twoLines)
        print("Line 1: " + str(angle1))
        print("Line 2: " + str(angle2))
        print("Angle between them: " + str(round(abs(angle1 - angle2))))

    cv.destroyAllWindows()
