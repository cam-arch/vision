import os
import pickle
import re
import timeit

import cv2 as cv
import natsort
import numpy as np


def image_pyramid(file, file_name, height, width):
    # Get image
    src = cv.imread(cv.samples.findFile(file))

    # Scale image down
    rows, cols, _channels = map(int, src.shape)  # get current image size
    while True:
        rows //= 2
        cols //= 2
        src = cv.pyrDown(src, dstsize=(rows, cols))  # scale image down

        # correct size found, break from loop
        if rows == height and cols == width:
            break

    # Give image black background
    (threshold, black_and_white_image) = cv.threshold(src, 127, 255, cv.THRESH_BINARY)

    name = format_name(file_name)
    return black_and_white_image, name


def format_name(name):
    name = name[4:]
    return name[:-4]


def get_templates(training_path, cache_path, use_cache):
    # Create folder for cached templates
    if os.path.exists(cache_path) and use_cache:
        print("Cache already exists. Using existing cache data.")
        return read_cache_templates(cache_path)

    # Create files for each template
    scales = [32, 64, 128, 256]  # The different scales we want to use
    rotations = [45, 90, 135, 180, 225, 270, 315]  # The rotations of the templates
    templates = {}
    for image in natsort.natsorted(os.listdir(training_path), reverse=False):
        for scale in scales:
            new_template, name = image_pyramid(training_path + "/" + image, image, scale, scale)
            templates[name + "_" + str(scale) + "_" + "0"] = new_template
            # Create rotated templates and add them to the dictionary
            for rotation in rotations:
                _, w, h = new_template.shape[::-1]
                rotation_angle = cv.getRotationMatrix2D((w / 2, h / 2), rotation, 1)  # Create rotation matrix
                new_template = cv.warpAffine(new_template, rotation_angle, (w, h))  # Rotate image
                templates[name + "_" + str(scale) + "_" + str(
                    rotation)] = new_template

    if use_cache:
        os.makedirs(cache_path)
        # Add dictionary to file
        with open(cache_path + "/cache.pkl", 'wb') as f:
            pickle.dump(templates, f)
            print("\nCached templates successfully.")

    return templates


def read_cache_templates(cache_path):
    # Read file
    with open(os.path.join(cache_path, "cache.pkl"), 'rb') as f:
        data = pickle.load(f)
        print("Read cache successfully.")
        return data


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


def run_task_2(training_path, test_path, test_annotations_path, cache_path, use_cache):
    # Create templates
    templates = get_templates(training_path, cache_path, use_cache)

    # Find templates in images
    templates_found = {}
    for image in natsort.natsorted(os.listdir(test_path), reverse=False):
        img_rgb = cv.imread(test_path + "/" + image)
        (threshold, black_and_white_image) = cv.threshold(img_rgb, 127, 255, cv.THRESH_BINARY)

        templates_found[image] = []
        for name, template in templates.items():
            _, w, h = template.shape[::-1]
            res = cv.matchTemplate(black_and_white_image, template, cv.TM_CCOEFF_NORMED)

            threshold = 0.9  # Making this 0.95 did not work for me
            loc = np.where(res >= threshold)
            if len(loc[0]) != 0:
                # TODO: This can sometimes be 1 pixel off
                y_min = round(np.average(loc[0]))
                x_min = round(np.average(loc[1]))
                y_max = y_min + w
                x_max = x_min + h
                templates_found[image].append((name, (x_min, y_min), (x_max, y_max)))
            for pt in zip(*loc[::-1]):
                cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        # cv.imshow('bbox', img_rgb)
        # cv.waitKey(0)
        print("\nChecked image: " + image)
        print(templates_found)

    accuracy_ = accuracy(test_annotations_path, templates_found)
    print("")
    print(accuracy_)

    cv.destroyAllWindows()


if __name__ == "__main__":
    training_path = "./Task2Dataset/Training/png"
    test_path = "./Task2Dataset/TestWithoutRotations/images"
    cache_path = training_path.replace("png", "cache")  # "./Task2Dataset/Training/cache"
    test_annotations_path = "./Task2Dataset/TestWithoutRotations/annotations"
    use_cache = True

    start = timeit.default_timer()
    run_task_2(training_path, test_path, test_annotations_path, cache_path, use_cache)
    stop = timeit.default_timer()
    print("Run-time: ", stop - start)
