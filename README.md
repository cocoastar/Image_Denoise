# Image_Denoise  
图像去噪的目标是从受噪声干扰的退化图像中尽可能恢复原始的真实图像，是图像进行后续处理的关键一步。在本文中，我们基于给定的REDNet和DnCNN两个典型深度图像去噪模型，在BSD68数据集上实现了图像去噪功能。基于课设要求，
## How to run  
### 环境配置要求  
* PyTorch
* OpenCV for Python
* HDF5 for Python
* Scikit-Image
### 运行训练函数  
```
python train.py \  
  --preprocess True \  
  --num_of_layers 17 \
  --out logs/DnCNN \  
  --noiseL 25 \  
  --val_noiseL 25
  --model DnCNN \
```
其中，preprocess参数在第一次训练时设置为True，它的作用是预处理训练集并生成h5文件，若文件目录下已存在该文件，则设置为False；num_of_layers为网络层数，可根据需要修改；out参数为训练模型参数所保存的位置，可根据自己的文件目录而定；noiseL为训练时所添加的噪声水平；model参数为训练的网络模型名称，可选DnCNN和REDNet。  
### 运行测试函数
```
python test.py \  
  --num_of_layers 17 \  
  --logdir logs/DnCNN \  
  --test_data Set12 \  
  --test_noiseL 15 \
  --model DnCNN  
```
注意，logdir为train函数保存的参数位置，test_data为测试数据集，可选Set12和Set68。
## 参考测试结果
### PSNR on Set12
 
 噪声水平 | REDNet10 | REDNet20 | REDNet30 | DnCNN-S
 ----- | ----- | ----- |----- | -----
 15  | 32.460 | 32.497 | 32.268 | 32.826 
 25  | 30.000 | 30.225 | 30.233 | 30.361  
 50  | 26.697 | 26.978 | 26.992 | 27.096
### PSNR on BSD68
