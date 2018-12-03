######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
TEST_MODE = False                           # turning on the testmode means the code will run on a small dataset.
CLEAN = True                               # set to "True" if you want to clean the temporary large files after generating result
MODEL = 'alexnet'                          # model arch: resnet18, alexnet, resnet50, densenet161
DATASET = 'imagenet'                       # model trained on: places365 or imagenet
QUANTILE = 0.005                            # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
CATAGORIES = ["object", "part","scene","texture","color"] # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
OUTPUT_FOLDER = "result/pytorch_"+MODEL+"_"+DATASET # result will be stored in this folder

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

MODEL_FILE = None
if MODEL != 'alexnet':
    DATA_DIRECTORY = 'dataset/broden1_224'
    IMG_SIZE = 224
else:
    DATA_DIRECTORY = 'dataset/broden1_227'
    IMG_SIZE = 227

if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000
if MODEL == 'resnet18':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = None
        MODEL_PARALLEL = False
elif MODEL.startswith('densenet'):
    FEATURE_NAMES = ['features']
    if DATASET == 'places365' and MODEL == 'densenet161':
        MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif MODEL.startswith('vgg'):
    FEATURE_NAMES = ['features']
elif MODEL.startswith('resnet'):
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365' and MODEL == 'resnet50':
        MODEL_FILE = 'zoo/whole_resnet50_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif MODEL == 'cornetz':
    FEATURE_NAMES = ['IT']
    if DATASET == 'imagenet':
        MODEL_FILE = 'zoo/cornet_z_epoch25.pth.tar'
        MODEL_PARALLEL = True
elif MODEL == 'cornetr':
    FEATURE_NAMES = ['IT']
    if DATASET == 'imagenet':
        MODEL_FILE = 'zoo/cornet_r_epoch25.pth.tar'
        MODEL_PARALLEL = True
elif MODEL == 'cornets':
    FEATURE_NAMES = ['IT']
    if DATASET == 'imagenet':
        MODEL_FILE = 'zoo/cornet_s_epoch43.pth.tar'
        MODEL_PARALLEL = True
elif MODEL == 'inception_v3':
    FEATURE_NAMES = ['Mixed_7c']
elif MODEL == 'xception':
    FEATURE_NAMES = ['conv4']
elif MODEL == 'pnasnet5large':
    FEATURE_NAMES = ['avg_pool']
elif MODEL == 'inceptionresnetv2':
    FEATURE_NAMES = ['avgpool_1a']
elif MODEL == 'nasnetalarge':
    FEATURE_NAMES = ['cell_17']
elif MODEL == 'nasnetamobile':
    FEATURE_NAMES = ['cell_15']
elif MODEL == 'alexnet':
    FEATURE_NAMES = ['features']
elif MODEL.startswith('squeezenet1'):
    FEATURE_NAMES = ['features']


if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 12
    BATCH_SIZE = 128
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4
    INDEX_FILE = 'index.csv'
