import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math

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


for image in sorted(os.listdir('./angle')):

    imageO = cv2.imread('./angle/' + image, cv2.IMREAD_GRAYSCALE)

    (thresh, blackAndWhiteImage) = cv2.threshold(imageO, 127, 255, cv2.THRESH_BINARY)

    # cv2.imshow('Image with Edge Detection', blackAndWhiteImage)


    # t2, thresh = cv2.threshold(imageO, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # t1 = 0.5 * t2

    # images.append(imageO)

    # image = cv2.Canny(imageO, t1, t2, None, 7)

    # # image = cv2.GaussianBlur(image, (5, 5), 2)

    # detected_edges_images.append(image)

    # # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(imageO, cv2.COLOR_GRAY2BGR)
    
    lines = cv2.HoughLines(blackAndWhiteImage, 1, np.pi / 155, 120, None, 0, 0)
    
    if lines is not None:
        twoLines = findTwoLines(lines)
        for idx in range(2):
            rho = twoLines[idx][0]
            theta = twoLines[idx][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    print("#######")

    cv2.imshow('Image with Edge Detection', cdst)

    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

    idx += 1


    







