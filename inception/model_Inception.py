import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionV3(nn.Module):
    def __init__(self, num_outputs=1, pretrained=True, dropout_p=0.5, task_type='regression'):
        super(InceptionV3, self).__init__()
        self.task_type = task_type

        self.backbone = inception_v3(weights=Inception_V3_Weights.DEFAULT if pretrained else None, aux_logits=True)

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

        if task_type == 'binary_classification':
            self.activation = nn.Sigmoid()
        elif task_type == 'multiclass_classification':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = nn.Identity()

        self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, tuple):
            x = x[0]
        return self.activation(x)

    def _initialize_weights(self):
        for m in self.backbone.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
