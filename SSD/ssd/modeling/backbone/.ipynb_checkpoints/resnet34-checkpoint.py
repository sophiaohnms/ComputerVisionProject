import torch
from torch import nn
from torchvision import models

original_model = models.resnet34(pretrained=True)

class ImprovedModel(torch.nn.Module):

    def __init__(self, cfg):

        super(ImprovedModel, self).__init__()

        self.features1 = nn.Sequential(
            # stop at last layer
            *list(original_model.children())[:-1])

        self.features2 = nn.Sequential(
            # stop at 2nd last layer
            *list(original_model.children())[:-2])

        self.features3 = nn.Sequential(
            # stop at 3rd last layer
            *list(original_model.children())[:-3])

        self.features4 = nn.Sequential(
            # stop at 4th last layer
            *list(original_model.children())[:-4])

        self.features5 = nn.Sequential(
            # stop at 5th last layer
            *list(original_model.children())[:-5])

        self.features6 = nn.Sequential(
            # stop at 6th last layer
            *list(original_model.children())[:-6])


    def forward(self, x):

        out1 = self.features1(x)
        out2 = self.features2(x)
        out3 = self.features3(x)
        out4 = self.features4(x)

        out_features = [out4, out3, out2, out1]

        #for idx, feature in enumerate(out_features):
            #print(feature.shape[1:])

        return tuple(out_features)

