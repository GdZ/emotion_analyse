# coding: utf-8
import numpy as np
# release
from utils import config
from utils.logger import logger


# ######################### load features & labels ######################
emotion = {'surprise': 0, 'anger': 1, 'happy': 2, 'love': 3, 'fear': 4, 'trust': 5, 'disgust': 6, 'sad': 7}
logger = logger(config)


def read_file(word_list, train_label, vector_text):
    feature_num = 0
    count = 0
    x = []
    y = []

    for line in word_list:
        feature_num += 1

    for line in train_label:
        y.append(line.strip())

    emotion_num = len(set(y))

    for line in vector_text:
        count += 1
        if not line.split():  # delete null string
            y[count - 1] = 'null'
            continue
        temp = list(line.strip().split(' '))
        t = list(map(int, temp))
        x.append(t)

    # delete label of null string
    y = [y[i] for i in range(len(y)) if y[i] != 'null']
    y = np.array(y[0: (count - 1)])

    return x, y, feature_num, emotion_num


# ######################## perception with sparse matrix(train) ################
# vector: list; weight: matrix
def matrix_dot(vector, weight):
    y_pre = np.zeros([1, weight.shape[1]])

    for i in vector:
        logger.d("weight -- len: %d, len(weight[i]): %s" % (len(weight), len(weight[i])))
        y_pre = y_pre + weight[i, :]

    return y_pre


def mod_weight(vector, weight, y_gold, y_pre):
    for i in vector:
        logger.i("mod_weight -- i: %d, y_gold:%d, y_pre:%d" %(i, y_gold, y_pre))
        weight[i, y_gold] += 1
        weight[i, y_pre] -= 1

    return weight


def train(word_list, train_label, vector_text, iteration):
    x, y, feature_num, emotion_num = read_file(word_list, train_label, vector_text)
    logger.i("@train() -- len(x):%d, len(y):%d, feature_num:%d, emotion_num:%d" \
            %(len(x), len(y), feature_num, emotion_num))

    w = np.zeros([feature_num, emotion_num])

    for i in range(iteration):
        logger.i("i:%d, iteration:%d" % (i, iteration))
        for m in range(len(x)):
            y_gold_str = y[m]
            y_gold = emotion[y_gold_str]
            y_pre = matrix_dot(x[m], w)
            logger.i("emotion: %s" % emotion)
            logger.i("y_gold_str, y_gold, y_pre = %s, %s, %s" %(y_gold_str, y_gold, str(y_pre)))
            re = np.where(y_pre == np.max(y_pre))
            logger.i("re: %s" % str(re))

            if len(re[1]) >= 2:
                y_pre = min(re[1])
            else:
                y_pre = re[1][0]

            if y_pre != y_gold:
                w = mod_weight(x[m], w, y_gold, y_pre)

    return x, y, w


# ######################## perception with sparse matrix(test) #################
def test(x, y, w):
    y_pre = []
    correct_num = 0

    for m in range(len(x)):
        y_pre = matrix_dot(x[m], w)
        re = np.where(y_pre == np.max(y_pre))
        if len(re[1]) >= 2:
            y_pre = min(re[1])
        else:
            y_pre = re[1][0]
        y_pre.append(y_pre)

    y_gold = list(y)
    for i in range(len(y_gold)):
        if emotion[y_gold[i]] == y_pre[i]:
            correct_num += 1
    accuracy = correct_num / len(y_gold)

    return accuracy
