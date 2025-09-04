
# import torch
# import torchvision.models as models
# import torch.nn as nn


# def get_vgg16(model_name, pretrained=False, num_classes=10, frozen=False):
#     if model_name=="vgg16":
#         model = VGG16Custom(pretrained=pretrained, num_classes=num_classes, frozen=frozen)
#         return model
#     else:
#         print("no such model")





# class VGG16Custom(nn.Module):
#     def __init__(self, pretrained=False, num_classes=10, frozen=False):
#         super(VGG16Custom, self).__init__()
#         # 加载预训练的 VGG16 模型
#         self.vgg16 = models.vgg16(pretrained=False)
#         if frozen:
#             for param in self.vgg16.parameters():
#                 param.requires_grad = False
#         # 修改 classifier 部分，添加两个全连接层
#         num_features = self.vgg16.classifier[6].in_features
#         self.vgg16.classifier = nn.Sequential(
#             nn.Linear(num_features, 4096),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         x = self.vgg16.features(x)
#         x = nn.AdaptiveAvgPool2d((1,1))(x)  # 调整特征图大小以适应全连接层
#         x = torch.flatten(x, 1)
#         x = self.vgg16.classifier(x)
#         return x
# '''VGG for CIFAR10. FC layers are removed.
# (c) YANG, Wei 
# '''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}





def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class VGG(nn.Module):

    def __init__(self, features, dataset_name="CIFAR10", num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier_0 = nn.Linear(512, 256)
        self.classifier_1 = nn.Linear(256, num_classes)
        self._initialize_weights()
        self.dataset_name=dataset_name
        print(dataset_name)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def forward(self, x, lin=0, lout=6,):
        
        if lout > 4:
            out = self.features(x)
            if self.dataset_name=="stl10":
                out = F.avg_pool2d(out,3)
            elif self.dataset_name=="tinyimagenet":
                out = F.avg_pool2d(out,2)  
        if lout >5 :
            out = out.view(out.size(0), -1)
            out = self.classifier_0(out)
            out = self.classifier_1(out)
        return out
        
        
        
        return x


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model
