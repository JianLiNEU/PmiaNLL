import torch
import torchvision.models as models
import torch.nn as nn

def get_inceptionv3(model_name, pretrained=False, num_classes=10, frozen=False):
    if model_name=="inceptionv3":
        model = inceptionv3Custom(pretrained=pretrained, num_classes=num_classes, frozen=frozen)
        return model
    else:
        print("no such model")



class inceptionv3Custom(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, frozen=False):
        super(inceptionv3Custom, self).__init__()
        # 加载预训练的 DenseNet121 模型
        self.inceptionv3 = models.inception_v3(pretrained=pretrained)
        if frozen:
            for param in self.inceptionv3.parameters():
                param.requires_grad = False
        # 修改 classifier 部分，添加两个全连接层
        num_features = self.inceptionv3.classifier[1].in_features
        
        self.linear = nn.Linear(num_features, num_classes)
        self.inceptionv3.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.inceptionv3(x)
    
    def forward(self, x, lin=0, lout=6,):
        if lout > 4:
            if self.dataset_name=="stl10":
                out = F.avg_pool2d(out, 8)
            else:
                out = F.avg_pool2d(out, 4)
        if lout >5:
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out   
        
        
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:#64*32*32
            out = self.layer1(out)
        if lin < 3 and lout > 1:#128*16*16
            out = self.layer2(out)
        if lin < 4 and lout > 2: #256*8*8
            out = self.layer3(out)
        if lin < 5 and lout > 3: #512*4*4
            out = self.layer4(out)
        if lout > 4:
            if self.dataset_name=="stl10":
                out = F.avg_pool2d(out, 8)
            else:
                out = F.avg_pool2d(out, 4)
        if lout >5:
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out