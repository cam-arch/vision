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

    imageO = cv2.imread('./angle/' + image, cv2.IMREAD_GRAYSCALE)

    (thresh, blackAndWhiteImage) = cv2.threshold(imageO, 127, 255, cv2.THRESH_BINARY)

    imageUnderlay = cv2.cvtColor(imageO, cv2.COLOR_GRAY2BGR)
    
    lines = cv2.HoughLines(blackAndWhiteImage, 1.1, np.pi / 150, 120, None, 0, 0)

    idx = 0
    
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

            if (idx == 0):
                cv2.line(imageUnderlay, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
                idx += 1
            else:
                cv2.line(imageUnderlay, pt1, pt2, (0,255,0), 1, cv2.LINE_AA)

        if (twoLines[0][0] >= 0):
            angle1 = 90 - round(np.degrees(twoLines[0][1]))
        else:
            angle1 = 270 - round(np.degrees(twoLines[0][1]))
            
        print ("Line 1: " + str(angle1))

        if (twoLines[1][0] >= 0):
            angle2 = 90 - round(np.degrees(twoLines[1][1]))
        else:
            angle2 = 270 - round(np.degrees(twoLines[1][1]))
            
        print ("Line 2: " + str(angle2))

        print (abs(angle1 - angle2))

    print("#######")

    cv2.imshow('Image with Edge Detection', imageUnderlay)

    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

    idx += 1


    







