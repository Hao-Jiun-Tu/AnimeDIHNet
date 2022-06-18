import numpy as np
import cv2 as cv
import os

dir = ['w_blur', 'wo_blur']

for i in range(2):
    real = cv.imread('{}/real_0000.png'.format(dir[i]), cv.IMREAD_UNCHANGED)
    comp = cv.imread('{}/comp_0000.png'.format(dir[i]), cv.IMREAD_UNCHANGED)
    mask = cv.imread('{}/mask_0000.png'.format(dir[i]), cv.IMREAD_UNCHANGED)
    result = cv.imread('{}/result_0000.png'.format(dir[i]), cv.IMREAD_UNCHANGED)

    cv.imwrite('{}/real_obj_0000.png'.format(dir[i]), real[95:415, 345:475])
    cv.imwrite('{}/comp_obj_0000.png'.format(dir[i]), comp[95:415, 345:475])
    cv.imwrite('{}/mask_obj_0000.png'.format(dir[i]), mask[95:415, 345:475])
    cv.imwrite('{}/result_obj_0000.png'.format(dir[i]), result[95:415, 345:475])
