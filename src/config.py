import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = int(os.environ.get('BATCH_SIZE', 512))
VAL_EVERY_N_EPOCH   = 1
NUM_EPOCHS          = 40

OPTIMIZER_TYPE      = os.environ.get('OPTIMIZER_TYPE', 'SGD')

if OPTIMIZER_TYPE=="Adagrad" or OPTIMIZER_TYPE=="Adam":
    # OPTIMIZER_PARAMS    = {'type': OPTIMIZER_TYPE, 'lr': 0.001}
    OPTIMIZER_PARAMS    = {'type': OPTIMIZER_TYPE, 'lr': 0.001, 'weight_decay': 1e-4} # weight decay 추가
else: # SGD or RMSProp
    OPTIMIZER_PARAMS    = {'type': OPTIMIZER_TYPE, 'lr': 0.005, 'momentum': 0.9}

SCHEDULER_TYPE      = os.environ.get('SCHEDULER_TYPE', 'MultiStepLR')

if SCHEDULER_TYPE=="MultiStepLR":
    SCHEDULER_PARAMS    = {'type': SCHEDULER_TYPE, 'milestones': [10, 20, 30], 'gamma': 0.2}

# Dataset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
MODEL_NAME          = os.environ.get('MODEL_NAME', 'resnet18')

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = os.environ.get('WANDB_NAME', f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}')
