# coding: utf-8
from utils.config import RELEASE as RELEASE
from utils.config import ORIGIN as ORIGIN


# prepare
TRAIN_CONTEXT_IDX = 4
SPLIT_STR = ';'

TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'

# setting for debug
if RELEASE:
    ROOT_PATH = './corpus/release/'

elif ORIGIN:
    ROOT_PATH = './corpus/old/'
    TRAIN_CONTEXT_IDX = 1
    SPLIT_STR = ' '
    TRAIN_FILE_NAME = 'train_c.csv'
    TEST_FILE_NAME = 'train_c.csv'

else:
    ROOT_PATH = './corpus/debug/'


# release data
TRAIN_CSV = ROOT_PATH + TRAIN_FILE_NAME
TEST_CSV = ROOT_PATH + TEST_FILE_NAME

# prepare
PROCESSING_TRAIN_FILE_NAME = 'processing_train.txt'
PROCESSING_TEST_FILE_NAME = 'processing_test.txt'
DEV_FILE_NAME = 'dev.csv'
PROCESSING_TRAIN_TXT = ROOT_PATH + PROCESSING_TRAIN_FILE_NAME
PROCESSING_TEST_TXT = ROOT_PATH + PROCESSING_TEST_FILE_NAME
DEV_FILE_CSV = ROOT_PATH + DEV_FILE_NAME

# word list vec
WORD_LIST_FILE_NAME = 'word_list.txt'
STOP_LIST_FILE_NAME = 'stop_word.txt'
WORD_LIST_TXT = ROOT_PATH + WORD_LIST_FILE_NAME
STOP_WORD_TXT = ROOT_PATH + STOP_LIST_FILE_NAME

# training
TRAIN_LABEL_FILE_NAME = 'train_label.csv'
TRAIN_LABEL_CSV = ROOT_PATH + TRAIN_LABEL_FILE_NAME

# perception
VECTOR_TRAIN_FILE_NAME = 'vector_train.txt'
VECTOR_TEST_FILE_NAME = 'vector_test.txt'
VECTOR_TRAIN_TXT = ROOT_PATH + VECTOR_TRAIN_FILE_NAME
VECTOR_TEST_TXT = ROOT_PATH + VECTOR_TEST_FILE_NAME

# bayes

# emb
EMB_FILE_NAME = 'emb_train.txt'
MODEL_FILE_NAME = 'model'
EMB_TRAIN_TXT = ROOT_PATH + EMB_FILE_NAME
OUTPUT_FILE_MODEL = ROOT_PATH + MODEL_FILE_NAME
