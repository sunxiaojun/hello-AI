# type: ignore
import numpy as np
import torch
import torch.nn as nn


class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        self.num_classes = num_classes
        # 第一层卷积
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 48, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # Downsampling 1/2
        )

        # 第二层卷积
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 128, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # Downsampling 1/4
        )

        # 第三层卷积
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 192, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # Downsampling 1/8
        )

        # 第四层卷积
        self.layer4 = nn.Sequential(
            nn.Conv2d(192, 256, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # Downsampling 1/16
        )

        # 第五层卷积
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, (3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # Downsampling 1/32
        )

        # 第六层使用卷积层取代FC层
        self.score_1 = nn.Conv2d(512, num_classes, (1, 1))
        self.score_2 = nn.Conv2d(256, num_classes, (1, 1))
        self.score_3 = nn.Conv2d(192, num_classes, (1, 1))

        # 第七层反卷积
        self.upsampling_2x = nn.ConvTranspose2d(num_classes, num_classes, (4, 4), (2, 2), (1, 1), bias=False)
        self.upsampling_4x = nn.ConvTranspose2d(num_classes, num_classes, (4, 4), (2, 2), (1, 1), bias=False)
        self.upsampling_8x = nn.ConvTranspose2d(num_classes, num_classes, (16, 16), (8, 8), (4, 4), bias=False)

        self._initialize_weights()

    @staticmethod
    def bilinear_kernel(in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                          dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight).float()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self.bilinear_kernel(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        h = self.layer2(h)
        s1 = self.layer3(h)  # 1/8
        s2 = self.layer4(s1)  # 1/16
        s3 = self.layer5(s2)  # 1/32

        s3 = self.score_1(s3)
        s3 = self.upsampling_2x(s3)
        s2 = self.score_2(s2)

        s2 = s2[:, :, :s3.size()[2], :s3.size()[3]]
        s2 += s3
        s2 = self.upsampling_4x(s2)
        s1 = self.score_3(s1)

        s1 = s1[:, :, :s2.size()[2], :s2.size()[3]]
        score = s1 + s2
        score = self.upsampling_8x(score)
        score = score[:, :, :x.size()[2], :x.size()[3]]

        return score

