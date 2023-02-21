import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import natsort

images = []
detected_edges_images = []
idx = 1

def findTwoLines(lines):

    maxDiff = 0
    maxDiffPos = 0
    cumRho1 = 0
    cumRho2 = 0
    cumTheta1 = 0
    cumTheta2 = 0

    sortedLines = sorted(lines, key=lambda x: x[0][0])

    for idx in range(len(sortedLines) - 1):

        if (sortedLines[idx + 1][0][0] - sortedLines[idx][0][0]) > maxDiff:

            maxDiffPos = idx + 1
            maxDiff = sortedLines[idx + 1][0][0] - sortedLines[idx][0][0]

    for jdx in range (0, maxDiffPos):

        cumRho1 += sortedLines[jdx][0][0]
        cumTheta1 += sortedLines[jdx][0][1]

    cumRho1 /= maxDiffPos
    cumTheta1 /= maxDiffPos

    for kdx in range (maxDiffPos, len(sortedLines)):
        
        cumRho2 += sortedLines[kdx][0][0]
        cumTheta2 += sortedLines[kdx][0][1]

    cumRho2 /= len(sortedLines) - maxDiffPos
    cumTheta2 /= len(sortedLines) - maxDiffPos

    twoLines = [[cumRho1, cumTheta1], [cumRho2, cumTheta2]]
    
    return twoLines


for image in natsort.natsorted(os.listdir('./angle'), reverse=False):

    image = cv2.imread('./angle/' + image, cv2.IMREAD_GRAYSCALE)

    t2, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    t1 = 0.5 * t2

    images.append(image)

    image = cv2.Canny(image, t1, t2, None, 7)

    # image = cv2.GaussianBlur(image, (5, 5), 2)

    detected_edges_images.append(image)

    # Copy edges to the images that will display the results in BGR
    imageUnderlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(image, 1.7, np.pi / 150, 100, None, 0, 0)

    if lines is not None:

        twoLines = findTwoLines(lines)

        for line in twoLines:

            rho = line[0]
            theta = line[1]

            a = math.cos(theta)
            b = math.sin(theta)

            x0 = a * rho
            y0 = b * rho

            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            cv2.line(imageUnderlay, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)

        print (abs(twoLines[0][1] - twoLines[1][1]) * 60), abs((twoLines[1][1] - twoLines[0][1]) * 60)

    print("#######")

    cv2.imshow('Image with Edge Detection', imageUnderlay)

    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

    idx += 1

