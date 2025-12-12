import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # 调用自定义的初始化函数
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 遍历模型的每一个模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 权重初始化为均值为0，标准差为0.01的高斯分布
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    # 根据AlexNet论文，偏置初始化
                    # 第2, 4, 5个卷积层偏置为1，其余卷积层偏置为0
                    # 注意：这里需要根据AlexNet的层顺序手动判断
                    # 鉴于Sequential没有直接暴露索引，我们可以根据输出通道数来间接判断
                    # 原始论文中的层编号是基于其顺序，这里我们假定按照PyTorch的Sequential顺序来判断
                    # Conv1 (out_channels=96): bias=0
                    # Conv2 (out_channels=256): bias=1
                    # Conv3 (out_channels=384): bias=0
                    # Conv4 (out_channels=384): bias=1
                    # Conv5 (out_channels=256): bias=1
                    if m.out_channels == 256 or (m.out_channels == 384 and m.in_channels != 256): # Conv2, Conv4, Conv5
                        nn.init.constant_(m.bias, 1)
                    else: # Conv1, Conv3
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                # 权重初始化为均值为0，标准差为0.01的高斯分布
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # 所有全连接层偏置为1
                nn.init.constant_(m.bias, 1)