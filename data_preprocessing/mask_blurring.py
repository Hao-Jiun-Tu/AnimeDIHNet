import numpy as np
import cv2 as cv
import os

input_path = '../video0/mask'
output_path = '../video0/blurred_mask'

if not os.path.exists(output_path):
    os.makedirs(output_path)

for maskname in os.listdir(input_path):
    filename = '{}/{}'.format(input_path, maskname)
    mask = cv.imread(str(filename), cv.IMREAD_UNCHANGED)
    # print('Image shape: {}'.format(img.shape))
    # print('Number of pixel 255: {}'.format(np.sum(img==255)))
    # cv.imshow('Mask', mask)
    # cv.waitKey()
    # cv.destroyAllWindows()
    
    blurred_mask = cv.GaussianBlur(mask, (3, 3), 0)
    # print('Number of pixel 255: {}'.format(np.sum(img==255)))
    # print('Total pixel: {}'.format(np.prod(img.shape)))  
    # cv.imshow('Blurred Mask', blurred_mask)
    # cv.waitKey()
    # cv.destroyAllWindows()
    
    cv.imwrite('{}/blurred_{}'.format(output_path, maskname), blurred_mask)
