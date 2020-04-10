import torch
from torch import nn



class BasicModel(torch.nn.Module):

    """
    This is a basic backbone for SSD.
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
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        image_channels = 3

        fc, pc = 3, 1
        fp, sp = 2, 2

        # Define the convolutional layers
        self.bank1 = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.MaxPool2d(kernel_size=fp, stride=sp),
            nn.ReLU(),                  #nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.MaxPool2d(kernel_size=fp, stride=sp),
            nn.ReLU(),                               #nn.BatchNorm2d(64),   #für andere implemntieren
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=self.output_channels[0],
                kernel_size=fc,
                stride=2,
                padding=pc
            ),
        )

        self.bank2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[0],
                out_channels=128,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=self.output_channels[1],
                kernel_size=fc,
                stride=2,
                padding=pc
            ),
        )

        self.bank3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[1],
                out_channels=256,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.output_channels[2],
                kernel_size=fc,
                stride=2,
                padding=pc
            ),
        )

        self.bank4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[2],
                out_channels=128,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=self.output_channels[3],
                kernel_size=fc,
                stride=2,
                padding=pc
            ),
        )

        self.bank5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[3],
                out_channels=128,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=self.output_channels[4],
                kernel_size=fc,
                stride=2,
                padding=pc
            ),
        )

        self.bank6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.output_channels[4],
                out_channels=128,
                kernel_size=(2,3),
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=fc,
                stride=1,
                padding=pc
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=self.output_channels[5],
                kernel_size=fc,
                stride=1,
                padding=0
            ),
        )

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """

        out1 = self.bank1(x)
        out2 = self.bank2(out1)
        out3 = self.bank3(out2)
        out4 = self.bank4(out3)
        out5 = self.bank5(out4)
        out6 = self.bank6(out5)

        out_features = [out1, out2, out3, out4, out5, out6]

        for idx, feature in enumerate(out_features):
            out_channel = self.output_channels[idx]
            feature_map_size = self.output_feature_size[idx]
                print(feature.shape[1:])
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            #assert feature.shape[1:] == expected_shape, \
                #f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

