PATCH_SIZE = 9
MAX_DISPARITY = 100

NUM_IMAGES = 200
NUM_TRAIN_IMAGES = 160
TRAIN_IMAGES_START_INDEX = 0
TRAIN_IMAGES_END_INDEX = NUM_TRAIN_IMAGES
VALIDATION_IMAGES_START_INDEX = NUM_TRAIN_IMAGES
VALIDATION_IMAGES_END_INDEX = NUM_IMAGES

# PUTANJE
__PROJECT_PATH = 'D:/FAKS ARHIVA/ZAVRSNI/Zavrsni-rad/'

__DISPARITY_DATA_ROOT_PATH = __PROJECT_PATH + 'disparity_data/'
TRAIN_DISPARITY_DATA_PATH = __DISPARITY_DATA_ROOT_PATH + 'train_disparity_data.npy'
VALIDATION_DISPARITY_DATA_PATH = __DISPARITY_DATA_ROOT_PATH + 'validation_disparity_data.npy'

__IMAGES_ROOT_PATH = __PROJECT_PATH + 'Kitti/stereo/'
DISPARITY_IMAGE_ROOT_PATH = __IMAGES_ROOT_PATH + 'disp_noc_0/'

COLOR_ROOT = __IMAGES_ROOT_PATH + 'color/'
GRAYSCALE_ROOT = __IMAGES_ROOT_PATH + 'grayscale/'

LEFT_ROOT = 'image_2/'
RIGHT_ROOT = 'image_3/'

COLOR_LEFT_ROOT = COLOR_ROOT + LEFT_ROOT
COLOR_RIGHT_ROOT = COLOR_ROOT + RIGHT_ROOT
GRAYSCALE_LEFT_ROOT = GRAYSCALE_ROOT + LEFT_ROOT
GRAYSCALE_RIGHT_ROOT = GRAYSCALE_ROOT + RIGHT_ROOT

TRAINED_MODEL_PATH = __PROJECT_PATH + 'trained_model_13.pth'

IS_GRAYSCALE = False
BATCH_NORMALIZATION = False
