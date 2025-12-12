import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_outputs=1, pretrained=True, dropout_p=0.5, task_type='regression'):
        super(ResNet50, self).__init__()
        self.task_type = task_type

        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = resnet50(weights=None)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_outputs)
        )

        # 激活函数
        if task_type == 'binary_classification':
            self.activation = nn.Sigmoid()
        elif task_type == 'multiclass_classification':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = nn.Identity()

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
        x = self.activation(x)
        return x

    def _initialize_weights(self):
        for m in self.backbone.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
