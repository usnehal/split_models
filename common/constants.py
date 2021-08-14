class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class test:
    STANDALONE = 1
    RGB_IMAGE_TRANSFER = 2
    RGB_IMAGE_TRANSFER_ZLIB = 3
    JPEG_TRANSFER = 4
    SPLIT_LAYER = 5
    SPLIT_LAYER_ZLIB = 6
    SPLIT_LAYER_QUANTIZED = 7
    SPLIT_LAYER_QUANTIZED_ZLIB = 8

class COM_REQUEST:
    LOAD_MODEL = 0
    PROCESS_JPEG_FILE = 1
    PROCESS_INT_TENSOR = 2

class BoxField:
    BOXES = 'bbox'
    KEYPOINTS = 'keypoints'
    LABELS = 'label'
    MASKS = 'masks'
    NUM_BOXES = 'num_boxes'
    SCORES = 'scores'
    WEIGHTS = 'weights'

class DatasetField:
    IMAGES = 'images'
    IMAGES_INFO = 'images_information'
    IMAGES_PMASK = 'images_padding_mask'

