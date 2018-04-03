# coding: utf-8
import numpy as np
# release
from utils import config
from utils.logger import logger
from utils.config import ORIGIN as ORIGIN

# ######################### load features & labels ######################
if ORIGIN:
    # original check_accuracy
    emotion = {'surprise': 0, 'anger': 1, 'happy': 2, 'love': 3, 'fear': 4, 'trust': 5, 'disgust': 6, 'sad': 7}
else:
    # self defined
    emotion = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, '8':7, '9':8, '10':9}

logger = logger(config)


# """
# parse vector_text to vectors and labels
# """
def parse(word_list, train_label, vector_text):
    feature_num = 0
    count = 0
    vectors = []
    labels = []

    for line in word_list:
        feature_num += 1

    for line in train_label:
        labels.append(line.strip())

    emotion_num = len(set(labels))

    for line in vector_text:
        logger.d('[ps->parse] line: %s' % line)
        if not line.split():  # delete null string
            logger.i('[ps->parse] it will delete null string.')
            labels[count] = 'null'
        temp = list(line.strip().split(' '))
        t = list(map(int, temp))
        vectors.append(t)
        count += 1

    # delete label of null string
    labels = [labels[i] for i in range(len(labels)) if labels[i] != 'null']
    labels = np.array(labels[0: count])
    logger.i('[Perception->parse] vectors:%d, labels:%d' % (len(vectors), len(labels)))

    return vectors, labels, feature_num, emotion_num


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
    vector, label, feature_num, emotion_num = parse(word_list, train_label, vector_text)
    logger.d("[ps->train] len(vectors):%d, len(y):%d, feature_num:%d, emotion_num:%d" \
             % (len(vector), len(label), feature_num, emotion_num))

    weight = np.zeros([feature_num, emotion_num])

    for i in range(iteration):
        logger.d("[ps->train] i:%d, x:%d, y:%d" % (i, len(vector), len(label)))
        for m in range(len(vector)):
            if len(label) <= m:
                logger.d('[ps->train] m is >= len(y)')
                continue
            y_gold_str = label[m]
            y_gold = emotion[y_gold_str]
            y_pre = matrix_dot(vector[m], weight)
            # logger.d("[ps->train] emotion: %s" % emotion)
            # logger.d("[ps->train] y_gold_str, y_gold, y_pre = %s, %s, %s" % (y_gold_str, y_gold, str(y_pre)))
            re = np.where(y_pre == np.max(y_pre))
            # logger.d("[ps->train] re: %s" % str(re))

            if len(re[1]) >= 2:
                y_pre = min(re[1])
            else:
                y_pre = re[1][0]

            if y_pre != y_gold:
                weight = mod_weight(vector[m], weight, y_gold, y_pre)

    return vector, label, weight


# generate labels by vectors and weights
def generate_labels(x, w):
    labels = [] # find the idx of max-value in each line(x)
    logger.i('[ps->generate_labels] x:%d, w:%d' %(len(x), len(w)))

    for m in range(len(x)):
        # y_pre = matrix_dot(x[m], w)
        tmp = matrix_dot(x[m], w)
        re = np.where(tmp == np.max(tmp))
        if len(re[1]) >= 2:
            tmp = min(re[1])
        else:
            tmp = re[1][0]
        # y_pre.append(tmp)
        logger.d('[ps->generate_labels] tmp: %s' % tmp)
        labels.append(tmp)

    return labels


# ######################## perception with sparse matrix(check_accuracy) #################
def check_accuracy(trained_labels, gold_labels, w):
    correct_num = 0
    logger.i('[ps->check_accuracy] pre_labels:%d, y:%d, w:%d' % (len(trained_labels), len(gold_labels), len(w)))

    y_gold = list(gold_labels)
    for i in range(len(y_gold)):
        if emotion[y_gold[i]] == trained_labels[i]:
            correct_num += 1
        else:
            logger.d('emotion[y_gold[%d]=%s]=%s and y_pre[%d]=%s'
                     % (i, y_gold[i], emotion[y_gold[i]], i, trained_labels[i]))
    # calculate the accuracy
    accuracy = (correct_num + 0.0) / len(y_gold) * 100.0
    logger.d('[Perception->check_accuracy] correct_num:%d, y_gold:%d = %f'
             % (correct_num, len(y_gold), accuracy))

    return accuracy
