import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math

images = []
detected_edges_images = []

t1 = 1
t2 = 200

idx = 1

for image in sorted(os.listdir('./angle')):

    image = cv2.imread('./angle/' + image, cv2.IMREAD_GRAYSCALE)

    images.append(image)

    new_image = cv2.Canny(image, t1, t2, None, 3)

    detected_edges_images.append(new_image)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
    
    lines = cv2.HoughLines(new_image, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            # print (lines)

            rho = lines[i][0][0]
            theta = lines[i][0][1]
            print ((theta * 360) / (2 * math.pi))
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow('Image with Edge Detection', cdst)

    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB))

    # plt.savefig('./images_detected_edges/image_' + str(idx))
    
    

    idx += 1







