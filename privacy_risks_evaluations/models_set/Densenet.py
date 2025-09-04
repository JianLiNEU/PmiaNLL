# import torch
# import torchvision.models as models
# import torch.nn as nn

# def get_densenet(model_name, pretrained=False, num_classes=10, frozen=False):
#     if model_name=="densenet121":
#         model = DenseNet121Custom(pretrained=pretrained, num_classes=num_classes, frozen=frozen)
#         return model
#     else:
#         print("no such model")



# class DenseNet121Custom(nn.Module):
#     def __init__(self, pretrained=False, num_classes=10, frozen=False):
#         super(DenseNet121Custom, self).__init__()
#         # 加载预训练的 DenseNet121 模型
#         if frozen:
#             for param in self.vgg19.parameters():
#                 param.requires_grad = False
#         self.densenet121 = models.densenet121(pretrained=pretrained)
#         # 修改 classifier 部分，添加两个全连接层
#         num_features = self.densenet121.classifier.in_features
#         self.densenet121.classifier = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         return self.densenet121(x)



import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['densenet']


from torch.autograd import Variable

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #print (x.shape)
        #print (self.bn1.num_groups)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, depth=100, block=Bottleneck,
        dropRate=0,dataset_name="cifar10", num_classes=10, growthRate=12, compressionRate=2):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2 
        if dataset_name == "MNIST":
            print("MNIST")
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                                bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)


    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.dense1(x)
    #     x = self.trans1(x)
    #     x = self.dense2(x)
    #     x = self.trans2(x)
    #     #x = self.trans1(self.dense1(x))
    #     #x = self.trans2(self.dense2(x))
    #     x = self.dense3(x)
    #     x = self.bn(x)
    #     x = self.relu(x)

    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)

    #     return x
    def forward(self, x, lin=0, lout=6,):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.dense1(out)
            out = self.trans1(out)
        if lin < 2 and lout > 0:#64*32*32
            out = self.dense2(out)
            out = self.trans2(out)
        if lin < 3 and lout > 1:#128*16*16
            out = self.dense3(out)
            out = self.bn(out)
            out = self.relu(out)
        if lout > 4:
            out = self.avgpool(out)
        if lout >5:
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        return out


def densenet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet(**kwargs)