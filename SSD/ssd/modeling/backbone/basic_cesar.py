import torch
from ssd import torch_utils


class BasicModelCesar(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature exztractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        
        
        #CNN STRUCTURE ___________________________________________________________________________________________
        self.l1 = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 32, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1), #ADDITION
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, output_channels[0], kernel_size=(3,3), stride=2, padding=1)
            #torch.nn.BatchNorm2d(output_channels[0])
        )

        self.l2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(output_channels[0]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels[0], 128, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1), #ADDITION
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1), #ADDITION
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1), #ADDITION
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, output_channels[1], kernel_size=(3,3), stride=2, padding=1)
            #torch.nn.BatchNorm2d(output_channels[1])
        )

        self.l3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(output_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels[1], 256, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, output_channels[2], kernel_size=(3,3), stride=2, padding=1)
            #torch.nn.BatchNorm2d(output_channels[2])
        )

        self.l4 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(output_channels[2]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels[2], 128, kernel_size=(2,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1), #ADDITION
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, output_channels[3], kernel_size=(3,3), stride=2, padding=1)
            #torch.nn.BatchNorm2d(output_channels[3])
        )

        self.l5 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(output_channels[3]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels[3], 128, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1), #ADDITION
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, output_channels[4], kernel_size=(3,3), stride=2, padding=1)
            #torch.nn.BatchNorm2d(output_channels[4])
        )

        self.l6 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(output_channels[4]),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels[4], 128, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1), #Add
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(), #Add
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            
            torch.nn.Conv2d(128, output_channels[5], kernel_size=(3,3), stride=1, padding=0)
            #torch.nn.BatchNorm2d(output_channels[5])    
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
        out_features = []
        
        
        #for idx, feature in enumerate(out_features):
        #    print(feature.shape[1:])
            
        #print("INPUT X: ", x.size())
        
        for idx, feature in enumerate(out_features):
            print(feature.shape[1:])
            
        x = self.l1(x)
        out_features.append(x)
        x = self.l2(x)
        out_features.append(x)
        x = self.l3(x)
        out_features.append(x)
        x = self.l4(x)
        out_features.append(x)
        x = self.l5(x)
        out_features.append(x)
        x = self.l6(x)
        out_features.append(x)
        
        
        #for idx, feature in enumerate(out_features):
            #print(feature.shape[1:])
            
        #print("OUT _0 ", out_features[0])
        
        """
        for idx, feature in enumerate(out_features):
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        """
        
        return tuple(out_features)



