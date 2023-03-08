import os
import pickle
import re
import timeit

import cv2 as cv
import natsort
import numpy as np
from sklearn.cluster import KMeans


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
    if os.path.exists(cache_path + "/cache.pkl") and use_cache:
        print("Cache already exists. Using existing cache data.")
        return read_cache_templates(cache_path)

    # Create files for each template
    scales = [64]  # The different scales we want to use
    rotations = []  # The rotations of the templates
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


# https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/#Implementing-IoU-using-NumPy
def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou


# from https://datascienceparichay.com/article/distance-between-two-points-python/
def get_distance(p, q):
    # sum of squared difference between coordinates
    s_sq_difference = 0
    for p_i, q_i in zip(p, q):
        s_sq_difference += (p_i - q_i) ** 2

    # take sq root of sum of squared difference
    distance = s_sq_difference ** 0.5
    return distance


def find_centre(template):
    x_diff = abs(template[1][0] - template[2][0])
    y_diff = abs(template[1][1] - template[2][1])

    return np.array([template[1][0] + x_diff // 2, template[1][1] + y_diff // 2])


def accuracy(test_annotations_path, icons_found):
    accuracy_ = {}
    annotation_pattern = r"^([a-zA-Z-]+), \((\d+), (\d+)\), \((\d+), (\d+)\)$"
    for image, templates_found in icons_found.items():
        templates_found = [[x[0].split('_')[0], x[1], x[2]] for x in templates_found]
        image = image.replace(".png", ".txt")

        with open(os.path.join(test_annotations_path, image), "r") as f:
            lines = f.readlines()
            correct_templates = []
            for i, line in enumerate(lines):
                match = re.match(annotation_pattern, line)
                if match:
                    result = [match.group(1), (int(match.group(2)), int(match.group(3))),
                              (int(match.group(4)), int(match.group(5)))]
                    correct_templates.append(result)

        templates_found_centres = np.empty((0, 2))
        for template_found in templates_found:
            templates_found_centres = np.append(templates_found_centres, [find_centre(template_found)], axis=0)

        correct_templates_centres = np.empty((0, 2))
        for correct_template in correct_templates:
            correct_templates_centres = np.append(correct_templates_centres, [find_centre(correct_template)], axis=0)

        kmeans = KMeans(n_clusters=len(correct_templates_centres), init=correct_templates_centres, max_iter=1)

        kmeans.fit(templates_found_centres)

        # At index i of labels, is index of our found templates. The value says which index of correct.
        labels = kmeans.predict(templates_found_centres)

        true_positive, false_positive = 0, 0
        # TODO: Change this - Need to get all the bboxes for each label and find the closest
        #  Our assumption is that this is our prediction is the closest
        for label, i in enumerate(labels):
            template_found = templates_found[i]
            correct_template = correct_templates[label]

            if template_found[0] == correct_template[0]:
                true_positive += 1
                iou = get_iou(
                    [template_found[1][0], template_found[1][1], template_found[2][0], template_found[2][1]],
                    [correct_template[1][0], correct_template[1][1], correct_template[2][0], correct_template[2][1]]
                )
            else:
                false_positive += 1

        false_negative = len(correct_templates) - true_positive - false_positive

    return accuracy_


def run_task_2(training_path, test_path, test_annotations_path, cache_path, use_cache):
    # Create templates
    templates = get_templates(training_path, cache_path, use_cache)

    # Find templates in images
    templates_found = {}
    count = 0
    for image in natsort.natsorted(os.listdir(test_path), reverse=False):
        if count == 1:
            break

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
                templates_found[image].append([name, (x_min, y_min), (x_max, y_max)])
            for pt in zip(*loc[::-1]):
                cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        # cv.imshow('bbox', img_rgb)
        # cv.waitKey(0)
        print("\nChecked image: " + image)
        print(templates_found)

        count += 1

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
