import os
import pickle
import re
import timeit

import cv2
import natsort
import numpy as np


def format_name(name):
    name = name[4:]
    return name[:-4]

def accuracy(test_annotations_path, templates_found):
    accuracy_ = {}
    annotation_pattern = r"^([a-zA-Z-]+), \((\d+), (\d+)\), \((\d+), (\d+)\)$"
    for image, template_found in templates_found.items():
        template_found = [(x[0].split('_')[0], x[1], x[2]) for x in template_found]
        image = image.replace(".png", ".txt")

        with open(os.path.join(test_annotations_path, image), "r") as f:
            lines = f.readlines()
            correct_template = []
            for i, line in enumerate(lines):
                match = re.match(annotation_pattern, line)
                if match:
                    result = (match.group(1), (int(match.group(2)), int(match.group(3))),
                              (int(match.group(4)), int(match.group(5))))
                    correct_template.append(result)

        # TODO: Split out if correct template found, and how close
        #  Euclidean distance of centre of boxes to determine
        #  Cam's method of how accurate (how much the bboxes overlap)
        correct, incorrect = 0, 0
        for template in template_found:
            if template in correct_template:
                correct += 1
            else:
                incorrect += 1

        accuracy_[image] = (correct, incorrect)

    return accuracy_


def run_task_3(training_path, test_path, test_annotations_path, cache_path, use_cache):

    for image in natsort.natsorted(os.listdir(test_path), reverse=False):
        img_rgb = cv2.imread(test_path + "/" + image)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp = sift.detect(gray,None)
        img=cv2.drawKeypoints(gray,kp,img)
        

    


if __name__ == "__main__":
    training_path = "./Task3AddtinalDataet/Training/png"
    test_path = "./Task3AddtinalDataet/TestWithoutRotations/images"
    cache_path = training_path.replace("png", "cache")  # "./Task3AddtinalDataet/Training/cache"
    test_annotations_path = "./Task3AddtinalDataet/TestWithoutRotations/annotations"
    use_cache = True

    start = timeit.default_timer()
    run_task_3(training_path, test_path, test_annotations_path, cache_path, use_cache)
    stop = timeit.default_timer()
    print("Run-time: ", stop - start)
