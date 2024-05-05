# paths
DATA_DF = './example_data_df.tsv'
DATASET_PATH = './example_dataset'
FEATURES_DF_PATH = './features_df.tsv'
SAVE_CLUSTERS = True
CLUSTERS_DIR = './clusters'

# img params
IMG_WIDTH = 1200
IMG_HEIGHT = 675
IMG_AREA = IMG_WIDTH * IMG_HEIGHT
SECTOR_AREA = 75 * 75

# clustering params
USE_IMAGE_FEATURES = True
IMG_EMBEDDER_MODEL = 'convnextv2_tiny.fcmae'
PCA_COMPONENTS = None
CLUSTERING_TYPE = 'optics'
MIN_SAMPLES = 30
PREDEFINED_CLUSTERS_COUNT = None
EPS = None
