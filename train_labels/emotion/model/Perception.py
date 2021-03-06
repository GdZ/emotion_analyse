# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
# release
from utils import config
from utils.logger import logger
from utils.config import ORIGIN as ORIGIN
from utils.config import DEBUG as DEBUG
from utils.config import RELEASE as RELEASE
# package
from emotion.model.Evaluation import Emotion

# ######################### load features & labels ######################
if ORIGIN:
    # original check_accuracy
    emotion = {'surprise': 0, 'anger': 1, 'happy': 2, 'love': 3, 'fear': 4, 'trust': 5, 'disgust': 6, 'sad': 7}
else:
    # self defined
    if RELEASE:
        emotion = {'-5':0, '-4':1, '-3':2, '-2':3, '-1':4, '0':5, '1':6, '2':7, '3':8, '4':9, '5':10}
        # emotion = {-5:0, -4:1, -3:2, -2:3, -1:4, 0:5, 1:6, 2:7, 3:8, 4:9, 5:10}
    else:
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

    # emotion_num = len(set(labels))
    emotion_num = len(emotion.values())

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
    y = [emotion[labels[i]] for i in range(len(labels)) if labels[i] != 'null']
    labels = np.array(labels[0: count])
    y = np.array(y[0: count])
    logger.i('[Perception->parse] vectors:%d, labels:%d' % (len(vectors), len(labels)))

    return vectors, labels, feature_num, emotion_num, y


# ######################## perception with sparse matrix(train) ################
# vector: list; weight: matrix
def matrix_dot(vector, weight):
    vec_hat = np.zeros([1, weight.shape[1]])

    for i in vector:
        # logger.d("[ps->matrix_dot] -- weight.len: %d, len(weight[%d]): %s" % (len(weight), i, len(weight[i])))
        vec_hat = vec_hat + weight[i, :]

    return vec_hat


def mod_weight(vector, weight, y, y_hat):
    for i in vector:
        logger.d("[ps->mod_weight] -- i:%d, y_gold:%d, y_pre:%d" % (i, y, y_hat))
        weight[i, y] += 1
        weight[i, y_hat] -= 1

    return weight


def train(word_list, train_label, vector_text, iteration):
    vector, label, feature_num, emotion_num, y = parse(word_list, train_label, vector_text)
    logger.d("[ps->train] len(vectors):{}, len(y):{}, feature_num:{}, emotion_num:{}"  \
                .format(len(vector), len(label), feature_num, emotion_num))

    weight = np.zeros([feature_num, emotion_num])

    for i in range(iteration):
        logger.d("[ps->train] i:%d, x:%d, y:%d" % (i, len(vector), len(label)))
        for m in range(len(vector)):
            if len(label) <= m:
                logger.d('[ps->train] m is >= len(y)')
                continue
            logger.d('[ps->train] label:[{}]: {}\nemotion: {}'.format(m, label, emotion))
            vec = vector[m]
            y_gold_str = label[m]
            y_gold = emotion[y_gold_str]

            y_hat = matrix_dot(vec, weight)

            # logger.d("[ps->train] emotion: %s" % emotion)
            # logger.d("[ps->train] y_gold_str, y_gold, y_pre = %s, %s, %s" % (y_gold_str, y_gold, str(y_pre)))
            re = np.where(y_hat == np.max(y_hat))
            # logger.d("[ps->train] re: %s" % str(re))

            if len(re[1]) >= 2:
                y_hat = min(re[1])
            else:
                y_hat = re[1][0]

            if y_hat != y_gold:
                weight = mod_weight(vec, weight, y_gold, y_hat)

    return vector, label, weight, y


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

def value_labels(val):
    labels = []
    keys = emotion.keys()
    values = emotion.values()
    for vv in val:
        labels.append(keys[values.index(vv)])
    return labels

# ######################## perception with sparse matrix(check_accuracy) #################
def check_accuracy(trained_labels, gold_labels, w):
    # correct_num = 0
    # logger.i('[Perception->check_accuracy] pre_labels:%d, y:%d, w:%d' % (len(trained_labels), len(gold_labels), len(w)))
    #
    # y_gold = list(gold_labels)
    # for i in range(len(y_gold)):
    #     if emotion[y_gold[i]] == trained_labels[i]:
    #         correct_num += 1
    #     else:
    #         logger.i('[Perception->check_accuracy] emotion[y_gold[{}]={}]={} and y_pre[{}]={}'.format(
    #                             i, y_gold[i], emotion[y_gold[i]], i, trained_labels[i]))
    # # calculate the accuracy
    # accuracy = (correct_num + 0.0) / len(y_gold) * 100.0
    # logger.d('[Perception->check_accuracy] correct_num:{}, y_gold:{} = {}'
    #          .format(correct_num, len(y_gold), accuracy))
    # logger.i('[Perception->check_accuracy] y_gold: {}, labels: {}'.format(len(y_gold), len(trained_labels)))
    # fig = plt.figure()
    # plt.plot(y_gold, trained_labels, 'r.')
    # plt.show()
    logger.i('[Perception->check_accuracy] gold_labels:{}'.format(gold_labels[:10]))
    logger.i('[Perception->check_accuracy] trained_labels:{}'.format(trained_labels[:10]))
    logger.i('[Perception->check_accuracy] emotion:{}'.format(emotion))
    evaluation = Emotion(gold_labels, trained_labels, emotion)
    evaluation.evaluation()
    # evaluation.print_result()
    # return accuracy
