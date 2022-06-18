# AnimeDIHNet
- Computational Photography Final Project : Animation Image Harmonization 
- Team Member: Jason Tu and Sharon Tsai
- Date: 2022/05/17-2022/6/17

## Requirements
- pytorch
- opencv
- numpy
- argparse
- imageio
- scikit-image

## Dataset
Since our topic is the specific application of image harmonization for animation images, and there is no related dataset supporting our demands, we need to construct animation image dataset by ourselves. You can also directly use [our dataset][1]. The following shows the our dataset construction flow:
![image0](experiment/image_for_readme/dataset_construction.png)
- Steps
    - Pick the target video to get animation images (`data_preprocessing/video_sub.py`).
    - Remove the background of ground truth images to get foreground masks, perform color transfer, and composite images (`data_preprocessing/data_preprocessing.py`).
    - (Optional) Blur foreground masks with Gaussian filter (`data_preprocessing/mask_blurring.py`)

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
                --nTest 20                                              \
                --cuda
```
Here we provide [three model files][2] that we had trained: `unet_mse.pth`, `unet_mse.pth` and `unet_attention.pth`.

### Experiments
## Foreground mask without/with blur
![image1](experiment/image_for_readme/experiment1.png)

## Patchsize and foreground object size
![image1](experiment/image_for_readme/experiment2.png)

## Comparison with recent work
![image1](experiment/image_for_readme/experiment3.png)
|          | fMSE      |    MSE    | PSNR      |    SSIM    |
| -------- | --------- |:---------:| --------- |:----------:|
| **Ours** | **62.17** | **11.83** | **35.83** | **0.9763** |
| iSSAM    | 98.59     |   13.47   | 34.60     |   0.9547   |

### Backup
- You can directly execute `run_train.bat` or `run_test.bat` to train or test the model. Some model arguments are written in the front part of `train.py` and `test.py`.
- There are more details in `document/`, and some reference papers are in `reference/`.

[1]:kkk
[2]:d