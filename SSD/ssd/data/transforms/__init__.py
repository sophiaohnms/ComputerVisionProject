from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *

#normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            
            #ADD SOME DATA AUGMENTATION
            RandomMirror(), #Augmentation
            #Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),  # "zoom-in" like in SSD paper
            ToPercentCoords(),
            #RandomBrightness(), #Basic transformations
            #RandomContrast(),
            #RandomSaturation(),
            #RandomHue(),
            # Albumentation Augmentations
            #add_weather(),
            #add_shadow(),
            #rotate(),
            #RandomLightingNoise(),
            #  # has to be before random sample crop
                                           # requires doubling training iterations
                                           # and is suggested by ssd paper to detect small objects
            #________________________________________
            Resize(cfg.INPUT.IMAGE_SIZE),
            #SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            Normalize([123, 117, 104], [0.229*255, 0.224*255, 0.225*255]),
            ToTensor(),

            # potential additional transformations
            # add rain etc from albumentation package
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            #SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            Normalize([123, 117, 104], [0.229*255, 0.224*255, 0.225*255]),
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
