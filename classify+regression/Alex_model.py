import torch.nn as nn
import torch
#N=（W-F+2*P）/S+1出现小数时自动删去补充的0行和0列

class AlexNet(nn.Module):
    def __init__(self, num_outputs=1, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1),  # stride=2, padding=1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # stride=2, pool尺寸3x3
            nn.Conv2d(48, 90, kernel_size=3, stride=2, padding=1),  # stride=2, padding=1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # stride=2, pool尺寸3x3
            nn.Conv2d(90, 128, kernel_size=3, padding=1),  # stride=1, padding=1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # stride=1, padding=1
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # stride=1, padding=1
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # stride=1, padding=1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # stride=2, pool尺寸3x3
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),#随机失活的比例
            nn.Linear(128 * 15 * 15, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_outputs),
        )
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)#第0维是batch，第1维是channel，从channel开始展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')#凯明初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)#正态分布初始化
                nn.init.constant_(m.bias, 0)
