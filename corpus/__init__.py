# coding: utf-8
from utils.config import RELEASE as RELEASE

# setting for debug
if RELEASE:
    ROOT_PATH = './corpus/release/'
else:
    ROOT_PATH = './corpus/debug/'

# prepare
TRAIN_CONTEXT_IDX = 4

# release data
TRAIN_CSV = ROOT_PATH + 'tweet.csv'
TEST_CSV = ROOT_PATH + 'test.csv'
# prepare
PROCESSING_TRAIN_TXT = ROOT_PATH + 'processing_train.txt'
PROCESSING_TEST_TXT = ROOT_PATH + 'processing_test.txt'
DEV_FILE_CSV = ROOT_PATH + 'dev.csv'
# word list vec
WORD_LIST_TXT = ROOT_PATH + 'word_list.txt'
STOP_WORD_TXT = ROOT_PATH + 'stop_word.txt'
# training
TRAIN_LABEL_CSV = ROOT_PATH + 'train_label.csv'
# perception
VECTOR_TRAIN_TXT = ROOT_PATH + 'vector_train.txt'
VECTOR_TEST_TXT = ROOT_PATH + 'vector_test.txt'
# bayes
# emb
EMB_TRAIN_TXT = ROOT_PATH + 'emb_train.txt'
