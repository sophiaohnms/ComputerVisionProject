import torch
from torch import nn
from torchvision import models

original_model = models.resnet34(pretrained=True)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class GivenModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        #print(original_model)

        fc, pc = 3, 1
        fm, sm = 2, 2

        self.resnetlayer0 = nn.Sequential(
            *list(original_model.children())[0:3]
        )

        self.resnetlayer1 = nn.Sequential(
            *list(original_model.children())[4:5]
        )
        # print("Layer 1: ", self.layer1)

        self.resnetlayer2 = nn.Sequential(
            *list(original_model.children())[5:6]
        )
        # print("Layer 2: ", self.layer2)

        self.resnetlayer3 = nn.Sequential(
                *list(original_model.children())[6:7]
        )

        self.layer1 = nn.Sequential(
            self.resnetlayer0,
            self.resnetlayer1,
            self.resnetlayer2,
            self.resnetlayer3,
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            BasicBlock(inplanes=256, planes = 512, stride = 2, downsample=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=fc,stride=2,padding=pc)),
            BasicBlock(inplanes=512, planes = 512)
        )

        self.layer3 = nn.Sequential(
            BasicBlock(inplanes=512, planes=256, stride = 2, downsample=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=fc,stride=2,padding=pc)),
            BasicBlock(inplanes=256, planes=256)
        )

        self.layer4 = nn.Sequential(
            BasicBlock(inplanes=256, planes=256, stride = 2, downsample=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=fc,stride=2,padding=pc)),
            BasicBlock(inplanes=256, planes=256)
        )

        self.layer5 = nn.Sequential(
            BasicBlock(inplanes=256, planes=128, stride = 2, downsample=nn.Conv2d(in_channels=256,out_channels=128,kernel_size=fc,stride=2,padding=pc)),
            BasicBlock(inplanes=128, planes=128)
        )

        self.layer6 = nn.Sequential(
            BasicBlock(inplanes=128, planes=128, stride = 2, downsample=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=fc,stride=2,padding=pc)),
            BasicBlock(inplanes=128, planes=128)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=fc,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )


    def forward(self, x):
        # print("Input: ", x.size())
        #out1 = self.resnetlayer0(x)
        #out2 = self.resnetlayer1(out1)
        #out3 = self.resnetlayer2(out2)
        l1 = self.layer1(x)
        #print("L1: ", l1.size())
        l2 = self.layer2(l1)
        #print("L2: ", l2.size())
        l3 = self.layer3(l2)
        #print("L3: ", l3.size())
        l4 = self.layer4(l3)
        #print("L4: ", l4.size())
        l5 = self.layer5(l4)
        #print("L5: ", l5.size())
        l6 = self.layer6(l5)
        #print("L6: ", l6.size())
        l7 = self.layer6(l6)
        #print("L7: ", l7.size())

        out_features = [l1, l2, l3, l4, l5, l7]

        for idx, feature in enumerate(out_features):
           print(feature.shape[1:])

        return tuple(out_features)

