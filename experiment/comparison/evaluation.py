import numpy as np
import cv2 as cv
import os
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric 

dir_list = ['./real_image/real_image', './blurred_mask/blurred_mask', './our_result/result', './issam_result/result']

psnr_result = 0
psnr_issam = 0
ssim_result = 0
ssim_issam = 0

mse_result = 0
mse_issam = 0

fmse_result = 0
fmse_issam = 0

for i in range(10):
    idx = '{:0>4}'.format(i)
    real_image_path = '{}_{}.png'.format(dir_list[0], idx)
    mask_path = '{}_{}.png'.format(dir_list[1], idx)
    result_path = '{}_{}.png'.format(dir_list[2], idx)
    issam_path = '{}_{}.png'.format(dir_list[3], idx)

    real_image = cv.imread(real_image_path, cv.IMREAD_UNCHANGED)
    mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
    result = cv.imread(result_path, cv.IMREAD_UNCHANGED)
    issam = cv.imread(issam_path, cv.IMREAD_UNCHANGED)

    mask = np.expand_dims(mask, 2)

    psnr_result += psnr_metric(real_image, result, data_range=255)
    ssim_result += ssim_metric(real_image, result, multichannel=True)
    psnr_issam += psnr_metric(real_image, issam, data_range=255)
    ssim_issam += ssim_metric(real_image, issam, multichannel=True)
    
    pixel_num = np.prod(real_image.shape)
    mse_result += np.sum(((real_image-result)**2)/pixel_num)
    mse_issam += np.sum(((real_image-issam)**2)/pixel_num)
    
    pixel_num = 3*np.sum(mask>=128)
    print(pixel_num)
    fmse_result += np.sum(((real_image-result)**2)/pixel_num)
    fmse_issam += np.sum(((real_image-issam)**2)/pixel_num)
    
print('--> result:\n Avg PSRN: {}\n Avg SSIM: {}\n MSE: {}\n fMSE: {}\n'.format(psnr_result/10, ssim_result/10, mse_result/10, fmse_result/10))
print('--> issam:\n Avg PSRN: {}\n Avg SSIM: {}\n MSE: {}\n fMSE: {}\n'.format(psnr_issam/10, ssim_issam/10, mse_issam/10, fmse_issam/10))