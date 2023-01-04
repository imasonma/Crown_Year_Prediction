import os
import yaml
from yacs.config import CfgNode as CN

######################
# config definition
######################
_C = CN()

########################
# Environment definition
########################
_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.CUDA = True
_C.ENVIRONMENT.SEED = 3407
_C.ENVIRONMENT.AMP_OPT_LEVEL = 'O1'
_C.ENVIRONMENT.NUM_GPU = 0
_C.ENVIRONMENT.MONITOR_TIME_INTERVAL = 0.1
_C.ENVIRONMENT.DATA_BASE_DIR = None
_C.ENVIRONMENT.PHASE = 'train'
_C.ENVIRONMENT.NAME = 'adam'
_C.ENVIRONMENT.EXPERIMENT_PATH = 'experiment'
_C.ENVIRONMENT.FINE_MODE = True

########################
# Dataset definition
########################
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 24
# num_classes is x + 1, when single-class is 0 + 1
_C.DATA.NUM_CLASSES = 0 + 1
_C.DATA.NUM_WORKERS = 12
_C.DATA.MAIN_MODAL = ['CT']
_C.DATA.SUB_MODAL = []
_C.DATA.LABEL = ['label']
_C.DATA.DIMENSION = 2
_C.DATA.IN_CHANNELS = 1
_C.DATA.NER_LAYER = 0
_C.DATA.FILE_PATH_TRAIN = '/raid0/myk/Y064/Dataset/NPY_train'
_C.DATA.FILE_PATH_VAL = '/raid0/myk/Y064/Dataset/NPY_val'
_C.DATA.SAVE = False
_C.DATA.SAVE_FILE = '/raid0/myk/AdaptationDataset/save/'

########################
# Model definition
########################
_C.MODEL = CN()
_C.MODEL.NAME = 'crossAttention'
_C.MODEL.TYPE = 'unet'
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.RESUME = ''
_C.MODEL.ZERO_TRAIN = False
_C.MODEL.TEST_MODE = False

########################
# Training definition
########################
_C.TRAINING = CN()
_C.TRAINING.START_EPOCH = 0
_C.TRAINING.EPOCH = 200
_C.TRAINING.LOSS = 'diceAndBce'
_C.TRAINING.ACTIVATION = 'sigmoid'
_C.TRAINING.METRIC = 'dice'
_C.TRAINING.LR_SCHEDULER = 'CosineLR'
_C.TRAINING.LOSS_SMOOTH = False
# OPTIMIZER config
_C.TRAINING.OPTIMIZER = CN()
_C.TRAINING.OPTIMIZER.METHOD = 'adam'
_C.TRAINING.OPTIMIZER.BASE_LR = 3e-4
_C.TRAINING.OPTIMIZER.WEIGHT_DECAY = 5e-5

# Test Time Adaptation
_C.TRAINING.TTA = CN()
_C.TRAINING.TTA.MODE = 'target'
_C.TRAINING.TTA.RESUME = ''
_C.TRAINING.TTA.UPDATE_PRARMS = 'all-BN'
_C.TRAINING.TTA.CONSISLEARNING = False
_C.TRAINING.TTA.BLOCK_BATCH = False

# Display the config
_C.DISPLAY = CN()
_C.DISPLAY.TRAIN = True
_C.DISPLAY.TRAIN_ITER = 100
_C.DISPLAY.TRAIN_IMAGE = True
_C.DISPLAY.TRAIN_LINE = True
_C.DISPLAY.VAL = True
_C.DISPLAY.VAL_ITER = 100
_C.DISPLAY.VAL_IMAGE = True
_C.DISPLAY.VAL_LINE = True

########################
# Data Augmentation definition
########################
_C.DATA_AUG = CN()
_C.DATA_AUG.IS_ENABLE = True
_C.DATA_AUG.IS_SHUFFLE = False
_C.DATA_AUG.IS_RANDOM_FLIP = False
_C.DATA_AUG.IS_RANDOM_ROTATE = False
_C.DATA_AUG.IS_RANDOM_ZOOM = False
_C.DATA_AUG.IS_RANDOM_GAUSS_SHARP = True
_C.DATA_AUG.IS_RANDOM_GAUSS_SMOTH = True
_C.DATA_AUG.IS_RANDOM_HIS_SHIFT = True
_C.DATA_AUG.IS_SHUFFLE_REMAP = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    # config.freeze()

def update_config(config, args):
    _update_config_from_file(config, args.cfg)


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
