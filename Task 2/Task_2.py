import os

import cv2 as cv
import natsort
import numpy as np

def image_pyramid(file, file_name, height, width):
    src = cv.imread(cv.samples.findFile(file))

    rows, cols, _channels = map(int, src.shape)
    rows //= 2
    cols //= 2

    while rows != 32 and cols != 32:
        src = cv.pyrDown(src, dstsize=(rows, cols))

        if (rows == height and cols == width):
            break
        
        # cv.imshow('Pyramids Demo', src)
        rows = rows // 2
        cols = cols // 2

        # cv.waitKey(0)

    (threshold, black_and_white_image) = cv.threshold(src, 127, 255, cv.THRESH_BINARY)

    return black_and_white_image, file_name


def run_task_2(training_path, test_path):
    scaled_templates = []
    # TODO: Cache templates
    #  Cache templates for sizes not 64 x 64
    for image in natsort.natsorted(os.listdir(training_path), reverse=False):
        scaled_template = image_pyramid(training_path + "/" + image, image, 64, 64)
        scaled_templates.append(scaled_template)

    templates_found = []
    for image in natsort.natsorted(os.listdir(test_path), reverse=False):
        img_rgb = cv.imread(test_path + "/" + image)
        (threshold, black_and_white_image) = cv.threshold(img_rgb, 127, 255, cv.THRESH_BINARY)

        templates_position = []
        for scaled_template, name in scaled_templates:
            # w, h = scaled_template.shape[::-1]
            # TODO: Make not 64 x 64
            w, h = 64, 64
            res = cv.matchTemplate(black_and_white_image, scaled_template, cv.TM_CCOEFF_NORMED)

            # Making this 0.95 did not work for me
            threshold = 0.9
            loc = np.where(res >= threshold)
            if len(loc[0]) != 0:
                # TODO: This can sometimes be 1 pixel off
                #  Also template has to be 64 x 64
                y_min = round(np.average(loc[0]))
                x_min = round(np.average(loc[1]))
                y_max = y_min + 64
                x_max = x_min + 64
                templates_position.append((name, (x_min, y_min), (x_max, y_max)))
            for pt in zip(*loc[::-1]):
                cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        templates_found.append(templates_position)
        # cv.imshow('bbox', img_rgb)
        # cv.waitKey(0)

    for images_templates in templates_found:
        print(images_templates)

    cv.destroyAllWindows()


if __name__ == "__main__":
    training_path = "./Task2Dataset/Training/png"
    test_path = "./Task2Dataset/TestWithoutRotations/images"
    try:    
        run_task_2(training_path, test_path)
    except:
        print("Laurence's laptop")
        run_task_2(training_path.replace(".", "Task 2"), test_path.replace(".", "Task 2"))
