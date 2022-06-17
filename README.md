# AnimeDIHNet

## Requirements
- pytorch
- opencv
- numpy
- argparse
- imageio
- scikit-image

## Usage
### Train
```
python train.py --nEpochs 240      \
                --cuda             \ 
                --thread 0
```
### Test
```
python test.py --model_path model_trained/<model_trained filename>.pth  \
                --nTest 20                                               \
                --cuda
```
Here we provide three model files that we had trained: `unet_mse.pth`, `unet_mse.pth` and `unet_attention.pth`.

### Backup
You can directly execute `run_train.bat` or `run_test` to train or test the model. Some model arguments are written in the front part of `train.py` and `test.py`.

