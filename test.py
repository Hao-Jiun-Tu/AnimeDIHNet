import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import imageio
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric 

# ===== Testing settings =====#
parser = argparse.ArgumentParser(description='NTHU EE - CP HW3 - ZebraSRNet')
parser.add_argument('--input_image_path', type=str, required=True, help='input image path')
parser.add_argument('--model_path', type=str, required=True, help='model file path')
parser.add_argument('--output_image_path', type=str, required=True, help='output image path')
parser.add_argument('--compare_image_path', type=str, help='ground-truth image to compare with the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
args = parser.parse_args()

print(args)

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

#===== load ZebraSRNet model =====#
print('===> Loading model')
net = torch.load(args.model_path)
if args.cuda:
    net = net.cuda()

#===== Load input image =====#
imgIn = imageio.imread(args.input_image_path)/255.0
imgIn = imgIn.transpose((2,0,1)).astype(float)
imgIn = imgIn.reshape(1, *imgIn.shape)
imgIn = torch.Tensor(imgIn)

#===== Test procedures =====#
varIn = Variable(imgIn)
if args.cuda:
    varIn = varIn.cuda()

prediction = net(varIn)
prediction = prediction.data.cpu().numpy().squeeze().transpose((1,2,0))
img_out = np.round(255*np.clip(prediction, 0.0, 1.0)).astype('uint8')
imageio.imwrite(args.output_image_path, img_out)

#===== Ground-truth comparison =====#
if args.compare_image_path is not None:
    imgTar = imageio.imread(args.compare_image_path)
    prediction = imageio.imread(args.output_image_path) 
    psnr = psnr_metric(imgTar, prediction, data_range=255)
    print('===> PSNR: {:.4f} dB'.format(psnr))
    ssim = ssim_metric(imgTar, prediction, multichannel=True)
    print('===> SSIM: {:.4f} dB'.format(ssim))