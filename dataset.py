import torch
import torch.utils.data as data
import random
import imageio
import os
import numpy as np

random.seed(777)
np.random.seed(777)
torch.manual_seed(777)

# ===== Utility functions for data augment =====#
def randomCrop(imgIn, imgTar, patchSize):
    (h, w, c) = imgIn.shape
    ix = random.randrange(0, w-patchSize+1)
    iy = random.randrange(0, h-patchSize+1)
    imgIn = imgIn[iy:iy+patchSize, ix:ix+patchSize, :]
    imgTar = imgTar[iy:iy+patchSize, ix:ix+patchSize, :]
    return imgIn, imgTar

def np2PytorchTensor(imgIn, imgTar):
    ts = (2, 0, 1)
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float))
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float))
    return imgIn, imgTar

def augment(imgIn, imgTar, hflip=True, vflip=True, rotation=True):
    if random.random() < 0.5 and hflip:
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]

    if random.random() < 0.5 and vflip:
        imgIn = imgIn[::-1, :, :]
        imgTar = imgTar[::-1, :, :]

    if random.random() < 0.5 and rotation:
        imgIn = imgIn.transpose(1, 0, 2)
        imgTar = imgTar.transpose(1, 0, 2)

    return imgIn, imgTar

# ===== Dataset =====#
class datasetTrain(data.Dataset):
    def __init__(self, args):
        self.patchSize = args.patchSize
        self.epochSize = args.epochSize
        self.batchSize = args.batchSize
        self.nTrain = args.nTrain
        self.trainDir = 'video0'
        self.imgInMaskPrefix = 'mask'
        self.imgInPrefix = 'composed'
        self.imgTarPrefix = 'real'
        with open(f'{self.trainDir}/random.txt', 'r') as f:
            self.dataset_samples = [int(x) for x in f.readlines()]

    def __getitem__(self, idx):
        idx = self.dataset_samples[(idx % self.nTrain)]

        nameIn_mask, nameIn, nameTar = self.getFileName(idx)
        imgIn_mask = imageio.imread(nameIn_mask)/255.0
        imgIn_mask = np.expand_dims(imgIn_mask, 2)
        
        imgIn = imageio.imread(nameIn)/255.0
        imgIn = np.concatenate((imgIn, imgIn_mask), axis=2)
        
        imgTar = imageio.imread(nameTar)/255.0
 
        imgIn, imgTar = randomCrop(imgIn, imgTar, self.patchSize)
        # imgIn, imgTar = augment(imgIn, imgTar)
        return np2PytorchTensor(imgIn, imgTar)

    def __len__(self):
        return self.epochSize*self.batchSize

    def getFileName(self, idx):
        fileName = '{:0>4}'.format(idx)
        nameIn_mask = '{}/{}_{}.png'.format(self.imgInMaskPrefix, self.imgInMaskPrefix, fileName)
        nameIn_mask = os.path.join(self.trainDir, nameIn_mask)
        
        nameIn = '{}_image/{}_{}.png'.format(self.imgInPrefix, self.imgInPrefix, fileName)
        nameIn = os.path.join(self.trainDir, nameIn)
        
        nameTar = '{}_image/{}_{}.png'.format(self.imgTarPrefix, self.imgTarPrefix, fileName)
        nameTar = os.path.join(self.trainDir, nameTar)

        return nameIn_mask, nameIn, nameTar

    
class datasetVal(data.Dataset):
    def __init__(self, args):
        self.patchSize = args.patchSize
        self.nVal = args.nVal
        self.valDir = 'video0'
        self.imgInMaskPrefix = 'mask'
        self.imgInPrefix = 'composed'
        self.imgTarPrefix = 'real'
        self.pad_size = (512, 1024)
        with open(f'{self.valDir}/random.txt', 'r') as f:
            self.dataset_samples = [int(x) for x in f.readlines()]

    def __getitem__(self, idx):
        n_sample = len(self.dataset_samples)-1
        idx = self.dataset_samples[n_sample-(idx % self.nVal)]

        nameIn_mask, nameIn, nameTar = self.getFileName(idx)
        imgIn_mask = imageio.imread(nameIn_mask)/255.0
        imgIn_mask = np.expand_dims(imgIn_mask, 2)
        imgIn_mask_padding = np.zeros((self.pad_size[0], self.pad_size[1], 1))
        imgIn_mask_padding[:imgIn_mask.shape[0], :imgIn_mask.shape[1], :] = imgIn_mask
        
        imgIn = imageio.imread(nameIn)/255.0
        imgIn_padding = np.ones((self.pad_size[0], self.pad_size[1], 3))
        imgIn_padding[:imgIn.shape[0], :imgIn.shape[1], :] = imgIn
        imgIn = np.concatenate((imgIn_padding, imgIn_mask_padding), axis=2)
        
        imgTar = imageio.imread(nameTar)/255.0
        
        #imgIn, imgTar = randomCrop(imgIn, imgTar, self.patchSize)
        return np2PytorchTensor(imgIn, imgTar)

    def __len__(self):
        return self.nVal

    def getFileName(self, idx):
        fileName = '{:0>4}'.format(idx)
        nameIn_mask = '{}/{}_{}.png'.format(self.imgInMaskPrefix, self.imgInMaskPrefix, fileName)
        nameIn_mask = os.path.join(self.valDir, nameIn_mask)
        
        nameIn = '{}_image/{}_{}.png'.format(self.imgInPrefix, self.imgInPrefix, fileName)
        nameIn = os.path.join(self.valDir, nameIn)
        
        nameTar = '{}_image/{}_{}.png'.format(self.imgTarPrefix, self.imgTarPrefix, fileName)
        nameTar = os.path.join(self.valDir, nameTar)

        return nameIn_mask, nameIn, nameTar
    
    
class datasetTest(data.Dataset):
    def __init__(self, args):
        self.nTest = args.nTest
        self.testDir = 'video0'
        self.imgInMaskPrefix = 'mask'
        self.imgInPrefix = 'composed'
        self.imgTarPrefix = 'real'
        self.pad_size = (512, 1024)
        with open(f'{self.testDir}/random.txt', 'r') as f:
            self.dataset_samples = [int(x) for x in f.readlines()]

    def __getitem__(self, idx):
        idx = self.dataset_samples[100+idx]

        nameIn_mask, nameIn, nameTar = self.getFileName(idx)
        img = imageio.imread(nameIn_mask)/255.0
        img = np.expand_dims(img, 2)
        imgIn_mask = np.zeros((self.pad_size[0], self.pad_size[1], 1))
        imgIn_mask[:img.shape[0], :img.shape[1], :] = img

        img = imageio.imread(nameIn)/255.0
        imgIn = np.ones((self.pad_size[0], self.pad_size[1], 3))
        imgIn[:img.shape[0], :img.shape[1], :] = img
        imgIn = np.concatenate((imgIn, imgIn_mask), axis=2)
        
        imgTar = imageio.imread(nameTar)
        return np2PytorchTensor(imgIn, imgTar)

    def __len__(self):
        return self.nTest

    def getFileName(self, idx):
        fileName = '{:0>4}'.format(idx)
        nameIn_mask = '{}/{}_{}.png'.format(self.imgInMaskPrefix, self.imgInMaskPrefix, fileName)
        nameIn_mask = os.path.join(self.testDir, nameIn_mask)
        
        nameIn = '{}_image/{}_{}.png'.format(self.imgInPrefix, self.imgInPrefix, fileName)
        nameIn = os.path.join(self.testDir, nameIn)
        
        nameTar = '{}_image/{}_{}.png'.format(self.imgTarPrefix, self.imgTarPrefix, fileName)
        nameTar = os.path.join(self.testDir, nameTar)

        return nameIn_mask, nameIn, nameTar
    