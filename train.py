import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from math import log10
# from model import AnimeDIHNet
# from model_att import AnimeDIHNet
from dataset import datasetTrain, datasetVal
import argparse
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from modified_loss import MaskWeightedMSE

#===== Training settings =====#
parser = argparse.ArgumentParser(description='NTHU EE - CP Final Project - AnimeDIHNet')
parser.add_argument('--patchSize', type=int, default=256, help='animation image cropping (patch) size for training')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--epochSize', type=int, default=150, help='number of batches as one epoch (for validating once)')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs for training')
parser.add_argument('--nTrain', type=int, default=400, help='number of training images')
parser.add_argument('--nVal', type=int, default=20, help='number of validation images')
parser.add_argument('--Loss', type=str, default='MSE', help='loss function: MSE or FN-MSE')
parser.add_argument('--attention', type=int, default=False, help='with/without attention layers: 1/0')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use, if Your OS is window, please set to 0')
parser.add_argument('--seed', type=int, default=777, help='random seed to use. Default=777')
parser.add_argument('--printEvery', type=int, default=30, help='number of batches to print average loss ')

args = parser.parse_args()
print(args)

print(torch.__version__)

if args.attention:
    from model_att import AnimeDIHNet
else:
    from model import AnimeDIHNet

if args.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.benchmark = False
    
#===== Datasets =====#
def seed_worker(worker_id):
    worker_seed = args.seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
print('===> Loading datasets')
train_set = datasetTrain(args)
train_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True, worker_init_fn=seed_worker)
val_set = datasetVal(args)
val_data_loader = DataLoader(dataset=val_set, num_workers=args.threads, batch_size=1, shuffle=False, worker_init_fn=seed_worker)

#===== AnimeDIHNet model =====#
print('===> Building model')
net = AnimeDIHNet()
# print('===> Loading model')
# net = torch.load('./model_trained/net_epoch_213.pth')
if args.cuda:
    net = net.cuda()

#===== Loss function and optimizer =====#
if args.Loss == 'MSE':
    criterion = torch.nn.MSELoss()
if args.Loss == 'FN-MSE':
    criterion = MaskWeightedMSE()

if args.cuda:
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

#===== Training and validation procedures =====#
def train(f, epoch):
    net.train()
    epoch_loss = 0
    for iteration, batch in enumerate(train_data_loader):
        varIn, varTar = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            varIn = varIn.cuda()
            varTar = varTar.cuda()

        optimizer.zero_grad()
        if args.Loss == 'MSE':
            loss = criterion(net(varIn), varTar)
        if args.Loss == 'FN-MSE':
            loss = criterion(net(varIn), varTar, varIn[:,3].unsqueeze(1))
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        if (iteration+1)%args.printEvery == 0:
            print("===> Epoch[{}]({}/{}): Avg. Loss: {:.4f}".format(epoch, iteration+1, len(train_data_loader), epoch_loss/args.printEvery))
            f.write("===> Epoch[{}]({}/{}): Avg. Loss: {:.4f}\n".format(epoch, iteration+1, len(train_data_loader), epoch_loss/args.printEvery))
            epoch_loss = 0

def validate(f):
    avg_psnr = 0
    avg_loss = 0
    mse_criterion = torch.nn.MSELoss()

    net.eval()
    with torch.no_grad():
        for batch in val_data_loader:
            varIn, varTar = Variable(batch[0]), Variable(batch[1])
            img_size = batch[1].shape
            if args.cuda:
                varIn = varIn.cuda()
                varTar = varTar.cuda()

            prediction = net(varIn)
            prediction[prediction > 1] = 1
            prediction[prediction < 0] = 0
            prediction = prediction[:, :, :img_size[2], :img_size[3]]
            mse = mse_criterion(prediction, varTar)
            print(mse.data)
            if args.Loss == 'MSE':
                loss = criterion(prediction, varTar).item()
            if args.Loss == 'FN-MSE':
                loss = criterion(prediction, varTar, varIn[:,3].unsqueeze(1)).item()
            psnr = 10 * log10(1.0*1.0/mse.item())
            print(psnr)
            avg_psnr += psnr
            avg_loss += loss
    avg_psnr /= len(val_data_loader)    
    avg_loss /= len(val_data_loader)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    f.write("===> Avg. PSNR: {:.4f} dB\n".format(avg_psnr))
    print("===> Avg. Loss: {:.4f}".format(avg_loss))
    f.write("===> Avg. Loss: {:.4f}\n".format(avg_loss))
    return avg_loss

#===== Model saving =====#
save_dir = './model_trained'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

def checkpoint(epoch): 
    save_name = 'net_epoch_{}.pth'.format(epoch)
    save_path = os.path.join(save_dir, save_name)
    torch.save(net, save_path)
    print("Checkpoint saved to {}".format(save_path))

#===== Main procedure =====#
with open('train_net.log', 'w') as f:
    f.write('training log record, random seed={}\n'.format(args.seed))
    f.write('dataset configuration: epoch size = {}, batch size = {}, patch size = {}\n'.format(args.epochSize, args.batchSize, args.patchSize))
    print('-------')

    for epoch in range(1, args.nEpochs+1):
        train(f, epoch)
        validate(f)
        checkpoint(epoch)
