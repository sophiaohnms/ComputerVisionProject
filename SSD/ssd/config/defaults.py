from yacs.config import CfgNode as CN

cfg = CN()

cfg.MODEL = CN()
cfg.MODEL.META_ARCHITECTURE = 'SSDDetector'
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
cfg.MODEL.THRESHOLD = 0.5
cfg.MODEL.NUM_CLASSES = 21
# Hard negative mining
cfg.MODEL.NEG_POS_RATIO = 3
cfg.MODEL.CENTER_VARIANCE = 0.1
cfg.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
cfg.MODEL.BACKBONE = CN()
cfg.MODEL.BACKBONE.NAME = 'basic'


#cfg.MODEL.BACKBONE.OUT_CHANNELS = (128, 256, 512, 512, 512, 128)
#cfg.MODEL.BACKBONE.OUT_CHANNELS = (512,1024,2048,2048) # resnet 50
cfg.MODEL.BACKBONE.OUT_CHANNELS = (256,512,256,256,128,128) # resnet 34 given


cfg.MODEL.BACKBONE.PRETRAINED = True
cfg.MODEL.BACKBONE.INPUT_CHANNELS = 3

# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------
cfg.MODEL.PRIORS = CN()

#cfg.MODEL.PRIORS.FEATURE_MAPS = [[40, 30], [20,15], [10,8], [5,4], [2,2], [1,1]] #basic
#cfg.MODEL.PRIORS.FEATURE_MAPS = [[40, 30], [20,15], [10,8], [5,4], [3,3], [1,1]] #basic

cfg.MODEL.PRIORS.FEATURE_MAPS = [[40, 30], [20,15], [10,8], [5,4], [3,2], [1,1]] # given model
cfg.MODEL.PRIORS.FEATURE_MAPS = [[40, 30], [20,15], [10,8], [5,4], [3,2], [2,1], [1,1]] # given model

#cfg.MODEL.PRIORS.FEATURE_MAPS = [[40, 30], [20,15], [10,8], [1,1]] #resnet50


#cfg.MODEL.PRIORS.STRIDES = [4, 8, 16, 32, 64, 100, 320] #Number of pixels between each box (In this case is: image size / feature map)
#cfg.MODEL.PRIORS.STRIDES = [8, 16, 32, 64, 100, 320]

cfg.MODEL.PRIORS.STRIDES = [[8,8], [16,16], [32,30], [64,60], [106,120], [160,240], [320, 240]] #for 3x2 box

#cfg.MODEL.PRIORS.STRIDES = [[8,8], [16,16], [32,30], [64,48], [160,120], [320, 240]] #for 2x2 box

cfg.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264] #We may want smaller/larger sizes. *Added 20,30 at the beginning
cfg.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]

cfg.MODEL.PRIORS.MIN_SIZES = [[32,24], [64,48], [118.4, 88.8], [172.8, 129.6], [227.2, 170.4], [227.2,211.2], [281.6, 211.2]] 
cfg.MODEL.PRIORS.MAX_SIZES = [[64,48], [118.4, 88.8], [172.8, 129.6], [227.2, 170.4], [281.6, 211.2],[300, 252], [336,252]]



#cfg.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2,3], [2, 3], [2, 3], [2, 3], [2], [2]] #Added [2] at the beginning
# When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
# #boxes = 2 + #ratio * 2
cfg.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]]


#cfg.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 6, 4, 4]  # number of boxes per feature map location. Added 4 at the beginning.
cfg.MODEL.PRIORS.BOXES_PER_LOCATION = [4, 6, 6, 6, 4, 4, 4]

cfg.MODEL.PRIORS.CLIP = True

# -----------------------------------------------------------------------------
# Box Head
# -----------------------------------------------------------------------------
cfg.MODEL.BOX_HEAD = CN()
cfg.MODEL.BOX_HEAD.NAME = 'SSDBoxHead'
cfg.MODEL.BOX_HEAD.PREDICTOR = 'SSDBoxPredictor'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
cfg.INPUT = CN()
# Image size
cfg.INPUT.IMAGE_SIZE = [320, 240] #*This has to be a tuple [320, 240], which is 4:3
# Values to be used for image normalization, RGB layout
cfg.INPUT.PIXEL_MEAN = [123, 117, 104]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
cfg.DATASETS = CN()
# List of the dataset names for training, as present in pathscfgatalog.py
cfg.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in pathscfgatalog.py
cfg.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
cfg.DATA_LOADER = CN()
# Number of data loading threads
cfg.DATA_LOADER.NUM_WORKERS = 16
cfg.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver - The same as optimizer
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
# train configs
cfg.SOLVER.MAX_ITER = 120000
cfg.SOLVER.LR_STEPS = [80000, 100000]
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.BATCH_SIZE = 32
cfg.SOLVER.LR = 1e-3
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 5e-4
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
cfg.SOLVER.WARMUP_ITERS = 500

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
cfg.TEST = CN()
cfg.TEST.NMS_THRESHOLD = 0.45
cfg.TEST.CONFIDENCE_THRESHOLD = 0.01
cfg.TEST.MAX_PER_CLASS = -1
cfg.TEST.MAX_PER_IMAGE = 100
cfg.TEST.BATCH_SIZE = 10

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
cfg.EVAL_STEP = 500 # Evaluate dataset every eval_step, disabled when eval_step < 0
cfg.MODEL_SAVE_STEP = 500 # Save checkpoint every save_step
cfg.LOG_STEP = 10 # Print logs every log_stepPrint logs every log_step
cfg.OUTPUT_DIR = "outputs"
cfg.DATASET_DIR = "datasets"
