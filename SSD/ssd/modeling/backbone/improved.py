import torch
from torch import nn
from torchvision import models

original_model = models.resnet152(pretrained=True)

class ImprovedModel(torch.nn.Module):

    """
    This is a resnet 152 backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """

    def __init__(self, cfg):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super(BasicModel, self).__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        image_channels = 3

        self.features1 = nn.Sequential(
            # stop at last layer
            *list(original_model.features.children())[:-1]

        self.features2 = nn.Sequential(
            # stop at 2nd last layer
            *list(original_model.features.children())[:-2]

        self.features3 = nn.Sequential(
            # stop at 3rd last layer
            *list(original_model.features.children())[:-3]

        self.features4 = nn.Sequential(
            # stop at 4th last layer
            *list(original_model.features.children())[:-4]

        self.features5 = nn.Sequential(
            # stop at 5th last layer
            *list(original_model.features.children())[:-5]

        self.features6 = nn.Sequential(
            # stop at 6th last layer
            *list(original_model.features.children())[:-6]
        )

    def forward(self, x):

        out1 = self.features1(x)
        out2 = self.features2(x)
        out3 = self.features3(x)
        out4 = self.features4(x)
        out5 = self.features5(x)
        out6 = self.features6(x)

        out_features = [out6, out5, out4, out3, out2, out1]

        for idx, feature in enumerate(out_features):
            print(feature.shape[1:])

        return tuple(out_features)

