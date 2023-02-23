import cv2 as cv


# 32 x 32 should be min size
def image_pyramid(file):
    print("""
    Zoom In-Out demo
    ------------------
    * [i] -> Zoom [i]n
    * [o] -> Zoom [o]ut
    * [ESC] -> Close program
    """)

    # Load the image
    src = cv.imread(cv.samples.findFile(file))
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: pyramids.py [image_name -- default ../data/chicky_512.png] \n')
        return -1

    while 1:
        rows, cols, _channels = map(int, src.shape)

        cv.imshow('Pyramids Demo', src)

        k = cv.waitKey(0)
        if k == 27:
            break

        elif chr(k) == 'i':
            src = cv.pyrUp(src, dstsize=(2 * cols, 2 * rows))
            print('** Zoom In: Image x 2')

        elif chr(k) == 'o':
            src = cv.pyrDown(src, dstsize=(cols // 2, rows // 2))
            print('** Zoom Out: Image / 2')

    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    image_pyramid("./Task2Dataset/Training/png/001-lighthouse.png")
