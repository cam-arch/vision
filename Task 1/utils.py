import numpy as np


def return_correct_angles_in_degrees(two_lines):
    if two_lines[0][0] >= 0:
        angle1 = 90 - np.degrees(two_lines[0][1])
    else:
        angle1 = 270 - np.degrees(two_lines[0][1])

    if two_lines[1][0] >= 0:
        angle2 = 90 - np.degrees(two_lines[1][1])
    else:
        angle2 = 270 - np.degrees(two_lines[1][1])

    return angle1, angle2
