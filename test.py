import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import imageio
import numpy as np
import argparse
from dataset import datasetTest
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric 

# ===== Testing settings =====#
parser = argparse.ArgumentParser(description='NTHU EE - CP Final Project - AnimeDIHNet')
parser.add_argument('--model_path', type=str, required=True, help='model file path')
parser.add_argument('--nTest', type=int, default=20, help='number of testing images')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
args = parser.parse_args()

print(args)

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

#===== AnimeDIHNet model =====#
print('===> Loading model')
net = torch.load(args.model_path)
if args.cuda:
    net = net.cuda()

#===== Datasets =====#
print('===> Loading datasets')
test_set = datasetTest(args)
test_data_loader = DataLoader(dataset=test_set)

#===== Testing procedures =====#
with open('test_result.log', 'w') as f:
    f.write('testing log record:')
    f.write('dataset size = {}\n'.format(args.nTest))
    f.write('------------------------------\n')
    for i, (imgIn, imgTar) in enumerate(test_data_loader):
        print('Image{}:'.format(i))
        f.write('Image{}:\n'.format(i))
        varIn = Variable(imgIn)
        img_real = imgTar.numpy().squeeze().transpose((1, 2, 0)).astype('uint8')
        img_size = img_real.shape
        if args.cuda:
            varIn = varIn.cuda()
 
        pred = net(varIn)
        pred = pred.data.cpu().numpy().squeeze().transpose((1, 2, 0))
        img_pred = np.round(255*np.clip(pred[:img_size[0], :img_size[1], :], 0, 1)).astype('uint8')
        imageio.imwrite('video0/test/result/result_{:0>4}.png'.format(i), img_pred)
        imageio.imwrite('video0/test/real_image/real_image_{:0>4}.png'.format(i), img_real)

        psnr = psnr_metric(img_real, img_pred, data_range=255)
        print('===> PSNR: {:.4f} dB'.format(psnr))
        f.write('===> PSNR: {:.4f} dB\n'.format(psnr))
        ssim = ssim_metric(img_real, img_pred, multichannel=True)
        print('===> SSIM: {:.4f} dB'.format(ssim))
        f.write('===> SSIM: {:.4f} dB\n'.format(ssim))
        print('------------------------------')
        f.write('------------------------------\n')
        
