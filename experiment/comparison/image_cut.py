import numpy as np
import cv2 as cv
import os

dir = 'image_cut'
comp = cv.imread('comp_image.png', cv.IMREAD_UNCHANGED)
our = cv.imread('our_result.png', cv.IMREAD_UNCHANGED)
issam = cv.imread('issam_result.png', cv.IMREAD_UNCHANGED)

num = 100
cv.imwrite('{}/comp_image.png'.format(dir), comp[:, num:])
cv.imwrite('{}/our_result.png'.format(dir), our[:, num:])
cv.imwrite('{}/issam_result.png'.format(dir), issam[:, num:])