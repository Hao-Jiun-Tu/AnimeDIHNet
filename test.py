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
    output_path = 'video0/test'
    avg_psnr = 0
    avg_ssim = 0
    avg_mse = 0
    avg_fmse = 0
    for i, (imgIn, imgTar) in enumerate(test_data_loader):
        print('Image{}:'.format(i))
        f.write('Image{}:\n'.format(i))
        
        varIn = Variable(imgIn)
        
        mask = imgIn[:,3].numpy().squeeze().transpose((0, 1))
        img_real = imgTar.numpy().squeeze().transpose((1, 2, 0)).astype('uint8')
        img_size = img_real.shape
        
        if args.cuda:
            varIn = varIn.cuda()
 
        pred = net(varIn)
        pred = pred.data.cpu().numpy().squeeze().transpose((1, 2, 0))
        img_pred = np.round(255*np.clip(pred[:img_size[0], :img_size[1], :], 0, 1)).astype('uint8')
        mask = np.round(255*np.clip(mask[:img_size[0], :img_size[1]], 0, 1)).astype('uint8')
        
        #===== Save predict images =====#
        imageio.imwrite('{}/result/result_{:0>4}.png'.format(output_path, i), img_pred)

        #===== Evaluation: 4 metrics =====#
        psnr = psnr_metric(img_real, img_pred, data_range=255)
        print('===> PSNR: {:.4f} dB'.format(psnr))
        f.write('===> PSNR: {:.4f} dB\n'.format(psnr))
        
        ssim = ssim_metric(img_real, img_pred, multichannel=True)
        print('===> SSIM: {:.4f} dB'.format(ssim))
        f.write('===> SSIM: {:.4f} dB\n'.format(ssim))

        mse = np.sum(((img_real-img_pred)**2)/np.prod(img_size))
        print('===> MSE: {:.4f}'.format(mse))
        f.write('===> MSE: {:.4f}\n'.format(mse))
        
        mask_size = 3*np.sum(mask>=128)
        fmse = np.sum(((img_real-img_pred)**2)/np.sum(mask_size))
        print('===> fMSE: {:.4f}'.format(fmse))
        f.write('===> fMSE: {:.4f}\n'.format(fmse))
        
        print('------------------------------')
        f.write('------------------------------\n')
        
        avg_psnr += psnr
        avg_ssim += ssim
        avg_mse += mse
        avg_fmse += fmse
        
    avg_psnr /= len(test_data_loader)
    avg_ssim /= len(test_data_loader)
    avg_mse /= len(test_data_loader)
    avg_fmse /= len(test_data_loader)
    print('===> Avg. PSNR: {:.4f} dB'.format(avg_psnr))
    f.write('===> Avg. PSNR: {:.4f} dB\n'.format(avg_psnr))
    print('===> Avg. SSIM: {:.4f}'.format(avg_ssim))
    f.write('===> Avg. SSIM: {:.4f}\n'.format(avg_ssim))
    print('===> Avg. MSE: {:.4f}'.format(avg_mse))
    f.write('===> Avg. MSE: {:.4f}\n'.format(avg_mse))
    print('===> Avg. fMSE: {:.4f}'.format(avg_fmse))
    f.write('===> Avg. fMSE: {:.4f}\n'.format(avg_fmse))