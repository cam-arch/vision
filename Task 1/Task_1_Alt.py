
import skimage
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import io

images = []
detected_edges_images = []

t1 = 1
t2 = 200

idx = 1

for image in sorted(os.listdir('./angle')):

    image = skimage.io.imread('./angle/' + image, as_gray=True)

    print ("TEST")

    # Compute the Canny filter 
    edges = skimage.feature.canny(image, sigma=3)

    images.append(edges)

    out, angles, d = skimage.transform.hough_line(edges)

    detected_edges_images.append(out)

    skimage.viewer.ImageViewer(out)

# display results
# skimage.viewer.CollectionViewer(images, update_on='move')

# skimage.viewer.CollectionViewer(detected_edges_images, update_on='move')

