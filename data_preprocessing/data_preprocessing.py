from rembg import remove
from PIL import Image
import os
import cv2
import random
from color_transfer import ColorTransfer
import numpy as np

input_path = '../video_data/video0'
output_path = '../animation_data/video0'

mask_path = '{}/mask'.format(output_path)
transfer_path = '{}/transfer'.format(output_path)
composed_path = '{}/composed_image'.format(output_path)
real_path = '{}/real_image'.format(output_path)

if not os.path.exists(mask_path):
        os.makedirs(mask_path)

if not os.path.exists(transfer_path):
        os.makedirs(transfer_path)

if not os.path.exists(composed_path):
        os.makedirs(composed_path)

if not os.path.exists(real_path):
        os.makedirs(real_path)

frame = 0
for image in os.listdir(input_path):
    data_name = '{}/{}'.format(input_path, image)
    print(data_name)
    ref_image = '{}/{}'.format(input_path, random.choice(os.listdir(input_path)))

    input = cv2.imread(data_name)
    img_arr_in = input
    img_arr_ref = cv2.imread(ref_image)

    cv2.imwrite('{}/real_{:0>4}.png'.format(real_path, frame), input)

    method = random.randint(0,1000) % 2
    # Initialize the class
    PT = ColorTransfer()

    if method == 0:
        # Pdf transfer
        img_trans = PT.pdf_transfer(img_arr_in=img_arr_in,
                                        img_arr_ref=img_arr_ref,
                                        regrain=True)
        cv2.imwrite(f'{transfer_path}/trans_{frame}.png', img_trans)
    else:
        # Lab mean transfer
        img_trans = PT.lab_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
        cv2.imwrite(f'{transfer_path}/trans_{frame}.png', img_trans)
    
    output = remove(input)

    b_channel, g_channel, r_channel, alpha= cv2.split(output)
    alpha[alpha>10] = 255
    alpha[alpha<=10] = 0
    mask = '{}/mask_{:0>4}.png'.format(mask_path, frame)
    cv2.imwrite(mask, alpha)

    alpha = alpha / 255
    alpha = np.expand_dims(alpha, 2).repeat(3, axis = 2)
    composed_img = input * (1-alpha) + img_trans * alpha
    composed_name = '{}/composed_{:0>4}.png'.format(composed_path, frame)
    cv2.imwrite(composed_name, composed_img)
    

    print(frame)
    frame += 1

