from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *

normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            
            #ADD SOME DATA AUGMENTATION
            RandomMirror(), #Augmentation
            Expand(cfg.INPUT.PIXEL_MEAN),
            ToPercentCoords(),
            RandomBrightness(), #Basic transformations
            RandomContrast(),
            RandomSaturation(),
            RandomHue(),
            #RandomLightingNoise(),
            #  # has to be before random sample crop
                                           # requires doubling training iterations
                                           # and is suggested by ssd paper to detect small objects
            RandomSampleCrop(), # "zoom-in" like in SSD paper
            #________________________________________
            Resize(cfg.INPUT.IMAGE_SIZE),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor(),

            # potential additional transformations
            # add rain etc from albumentation package
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
