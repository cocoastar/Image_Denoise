import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AvgPool2d(stride=2,kernel_size=2))#下采样
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        '''
        layers.append(nn.Conv2d(in_channels=features, out_channels=4*channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.PixelShuffle(2))
        '''
        layers.append(nn.Upsample(mode='bicubic',scale_factor=2))
        layers.append(nn.Conv2d(features,channels,kernel_size=3,stride=1,padding=1))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
