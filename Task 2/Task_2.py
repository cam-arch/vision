import os

import cv2 as cv
import natsort
import json
import numpy as np

def image_pyramid(file, file_name, height, width):
    ## Get image
    src = cv.imread(cv.samples.findFile(file))

    ## Scale image down
    rows, cols, _channels = map(int, src.shape)         # get current image size
    while rows != 32 and cols != 32:
        rows //= 2
        cols //= 2
        src = cv.pyrDown(src, dstsize=(rows, cols))     # scale image down

        if (rows == height and cols == width): break    # correct size found; break from loop

    ## Give image black background
    (threshold, black_and_white_image) = cv.threshold(src, 127, 255, cv.THRESH_BINARY)

    return black_and_white_image, file_name


def create_cache_templates(training_path, cache_path):
    ## Create folder for cached templates
    if os.path.exists(cache_path):
        print("Cache already exists. Using old cache data.")
        return
    else:
        os.makedirs(cache_path)
    
    ## Create files for each template
    scales = [64, 128]      # The differenet scales we want to use
    rotations = [45, 90, 135, 180, 225, 270, 315]   # The rotations of the templates
    for image in natsort.natsorted(os.listdir(training_path), reverse=False):
        templates = {}

        ## Create scaled templates and add them to the dictionary
        scaled_templates = []
        for scale in scales:
            new_template, name = image_pyramid(training_path + "/" + image, image, scale, scale)
            scaled_templates.append(new_template)
            templates[name + str(scale)] = new_template.tolist()

        ## Create rotated templates and add them to the dictionary
        for rotation in rotations:
            for scaled_template in scaled_templates:
                _, w, h = scaled_template.shape[::-1]
                rotation_angle = cv.getRotationMatrix2D((w/2, h/2), rotation, 1)            # Create rotation matrix
                new_template = cv.warpAffine(scaled_template, rotation_angle, (w, h))       # Rotate image
                templates[name + str(rotation)] = new_template.tolist()                                 # Add image to dictionary. tolist() needed for json serializing

        ## Add dictionary to file
        with open(cache_path + "/" + str(image).replace("png", "txt"), 'w') as f:
            json.dump(dict(templates), f)
        f.close()

    print("\nCached all templates successfuly.")


def read_cache_templates(cache_path):
    templates = []
    for filename in natsort.natsorted(os.listdir(cache_path), reverse=False):
        ## Read file
        with open(os.path.join(cache_path, filename), 'r') as f:
            data = json.load(f)
        
        ## Convert dictionary of images to one long list of images
        images = list(data.values())
        for image in images:
            image = np.array(image)             # Convert list to numpy.
            image = image.astype(np.uint8)      # Convert numpy dtype to 8-bit. Needed for opencv
            templates.append((image, filename[:-3]))

    print("Read cache successfuly.")
    return templates


def run_task_2(training_path, test_path, cache_path, use_cache):
    ## Test path because of Laurence's laptop
    x = os.listdir(training_path)

    ## Create templates
    templates = []
    if (use_cache):
        create_cache_templates(training_path, cache_path)      # Create cache templates
        templates = read_cache_templates(cache_path)        # Read cache templates
    else:
        for image in natsort.natsorted(os.listdir(training_path), reverse=False):
            template = image_pyramid(training_path + "/" + image, image, 64, 64)
            templates.append(template)
   

    ## Find templates in images
    templates_found = []
    for image in natsort.natsorted(os.listdir(test_path), reverse=False):
        img_rgb = cv.imread(test_path + "/" + image)
        (threshold, black_and_white_image) = cv.threshold(img_rgb, 127, 255, cv.THRESH_BINARY)

        templates_position = []
        for template, name in templates:
            _, w, h = template.shape[::-1]
            res = cv.matchTemplate(black_and_white_image, template, cv.TM_CCOEFF_NORMED)

            threshold = 0.9     # Making this 0.95 did not work for me
            loc = np.where(res >= threshold)
            if len(loc[0]) != 0:
                # TODO: This can sometimes be 1 pixel off
                #  Also template has to be 64 x 64
                y_min = round(np.average(loc[0]))
                x_min = round(np.average(loc[1]))
                y_max = y_min + w
                x_max = x_min + h
                templates_position.append((name, (x_min, y_min), (x_max, y_max)))
            for pt in zip(*loc[::-1]):
                cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        templates_found.append(templates_position)
        # cv.imshow('bbox', img_rgb)
        # cv.waitKey(0)
        print("\nChecked image: " + image)
        print(templates_found)

    ## Print output
    for images_templates in templates_found:
        print(images_templates)

    cv.destroyAllWindows()


if __name__ == "__main__":
    training_path = "./Task2Dataset/Training/png"
    test_path = "./Task2Dataset/TestWithoutRotations/images"
    cache_path = training_path.replace("png", "cache")      # "./Task2Dataset/Training/cache"
    use_cache = True

    try:    
        run_task_2(training_path, test_path, cache_path, use_cache)
    except:
        print("Laurence's laptop")
        run_task_2(training_path.replace(".", "Task 2"), test_path.replace(".", "Task 2"), cache_path.replace(".", "Task 2"), use_cache)
