import torch
from torch import nn
from torchvision import models

resnet = models.resnet50(pretrained=True)


class ResNet50(torch.nn.Module):

    def __init__(self, cfg):
        super(ResNet50, self).__init__()
        #print(resnet)

        self.layer1 = nn.Sequential(
            *list(resnet.children())[0:6]
        )
        self.layer2 = nn.Sequential(
            *list(resnet.children())[6:7]
        )
        # print("Layer 2: ", self.layer2)

        self.layer3 = nn.Sequential(
            *list(resnet.children())[7:8]
        )
              
        self.layer4 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(2048, 2048, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(),
            nn.Conv2d(2048, 2048, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # print("Layer 4: ", self.layer4)

        self.layer5 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=(3,2), stride=1, padding=1), #From [5,4] to [3,3]
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(1024, 1024, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #
        # print("Layer 5: ", self.layer5)

        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(3,3), stride=1, padding=1), #From [5,4] to [3,3]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        # self.layer6 = nn.Sequential(
        #    nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # )

        # print("Layer 6: ", self.layer6)

    def forward(self, x):
        #print("THIS IS RESNET 50")
        #print("Input: ", x.size())
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

        out_features = [l1, l2, l3, l4, l5, l6]

        # for idx, feature in enumerate(out_features):
        #    print(feature.shape[1:])

        return tuple(out_features)