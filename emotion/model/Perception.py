# coding: utf-8
import numpy as np
# release
from utils import config
from utils.logger import logger
from utils.config import ORIGIN as ORIGIN

# ######################### load features & labels ######################
if ORIGIN:
    # original test
    emotion = {'surprise': 0, 'anger': 1, 'happy': 2, 'love': 3, 'fear': 4, 'trust': 5, 'disgust': 6, 'sad': 7}
else:
    # self defined
    emotion = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9}

logger = logger(config)


def parse(word_list, train_label, vector_text):
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
        logger.d('[ps->parse] line: %s' % line)
        if not line.split():  # delete null string
            logger.i('[ps->parse] it will delete null string.')
            y[count] = 'null'
        temp = list(line.strip().split(' '))
        t = list(map(int, temp))
        x.append(t)
        count += 1

    # delete label of null string
    y = [y[i] for i in range(len(y)) if y[i] != 'null']
    y = np.array(y[0: count])
    logger.i('[Perception->parse] x:%d, y:%d' % (len(x), len(y)))

    return x, y, feature_num, emotion_num


# ######################## perception with sparse matrix(train) ################
# vector: list; weight: matrix
def matrix_dot(vector, weight):
    y_pre = np.zeros([1, weight.shape[1]])

    for i in vector:
        # logger.d("[ps->matrix_dot] -- weight.len: %d, len(weight[%d]): %s" % (len(weight), i, len(weight[i])))
        y_pre = y_pre + weight[i, :]

    return y_pre


def mod_weight(vector, weight, y_gold, y_pre):
    for i in vector:
        logger.d("[ps->mod_weight] -- i:%d, y_gold:%d, y_pre:%d" % (i, y_gold, y_pre))
        weight[i, y_gold] += 1
        weight[i, y_pre] -= 1

    return weight


def train(word_list, train_label, vector_text, iteration):
    x, y, feature_num, emotion_num = parse(word_list, train_label, vector_text)
    logger.d("[ps->train] len(x):%d, len(y):%d, feature_num:%d, emotion_num:%d" \
             % (len(x), len(y), feature_num, emotion_num))

    w = np.zeros([feature_num, emotion_num])

    for i in range(iteration):
        # logger.i("[ps->train] i:%d, x:%d, y:%d" % (i, len(x), len(y)))
        for m in range(len(x)):
            if len(y) <= m:
                logger.w('[ps->train] m is >= len(y)')
                continue
            y_gold_str = y[m]
            y_gold = emotion[y_gold_str]
            y_pre = matrix_dot(x[m], w)
            # logger.d("[ps->train] emotion: %s" % emotion)
            # logger.d("[ps->train] y_gold_str, y_gold, y_pre = %s, %s, %s" % (y_gold_str, y_gold, str(y_pre)))
            re = np.where(y_pre == np.max(y_pre))
            # logger.d("[ps->train] re: %s" % str(re))

            if len(re[1]) >= 2:
                y_pre = min(re[1])
            else:
                y_pre = re[1][0]

            if y_pre != y_gold:
                w = mod_weight(x[m], w, y_gold, y_pre)

    return x, y, w


# ######################## perception with sparse matrix(test) #################
def test(x, y, w):
    y_pre = [] # find the idx of max-value in each line(x)
    correct_num = 0
    logger.i('[ps->test] x:%d, y:%d, w:%d' %(len(x), len(y), len(w)))

    for m in range(len(x)):
        # y_pre = matrix_dot(x[m], w)
        tmp = matrix_dot(x[m], w)
        re = np.where(tmp == np.max(tmp))
        if len(re[1]) >= 2:
            tmp = min(re[1])
        else:
            tmp = re[1][0]
        # y_pre.append(tmp)
        y_pre.append(tmp)

    y_gold = list(y)
    for i in range(len(y_gold)):
        if emotion[y_gold[i]] == y_pre[i]:
            correct_num += 1
        else:
            logger.i('emotion[y_gold[%d]=%s]=%s and y_pre[%d]=%s' % (i, y_gold[i], emotion[y_gold[i]], i, y_pre[i]))
    accuracy = (correct_num + 0.0) / len(y_gold)
    logger.d('[Perception->test] correct_num:%d, y_gold:%d = %f' % (correct_num, len(y_gold), accuracy))

    return accuracy
