import torch
import torchvision.models as models
import torch.nn as nn

def get_mobilenetv2(model_name, pretrained=False, num_classes=10, frozen=False):
    if model_name=="mobilenetv2":
        model = Mobilenetv2Custom(pretrained=pretrained, num_classes=num_classes, frozen=frozen)
        return model
    else:
        print("no such model")



class Mobilenetv2Custom(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, frozen=False):
        super(Mobilenetv2Custom, self).__init__()
        # 加载预训练的 DenseNet121 模型
        self.mobilenetv2 = models.mobilenet_v2(pretrained=pretrained)
        if frozen:
            for param in self.mobilenetv2.parameters():
                param.requires_grad = False
        # 修改 classifier 部分，添加两个全连接层
        num_features = self.mobilenetv2.classifier[1].in_features
        self.mobilenetv2.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.mobilenetv2(x)