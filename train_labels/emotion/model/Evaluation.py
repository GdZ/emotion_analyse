#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:08:06 2017

@author: Mengru Ding, Zhanruo Qu, Helen Vernon
"""
import numpy as np
from utils import config
from utils.logger import logger

logger = logger(config)


class Emotion:
    def __init__(self, y_labels, y_predict, emotion):
        self.emotion = emotion
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.F1 = 0
        self.gold_labels = y_labels
        self.predictions = y_predict

    def evaluation(self):
        labels = list(self.emotion.keys())
        values = list(self.emotion.values())
        self.precision = self.recall = self.F1 = np.zeros(np.shape(self.emotion))
        # for each emotion
        for idx, e in enumerate(self.emotion):
            self.tp = self.tn = self.fp = self.fn = 0
            self.accuracy = 0.
            for i in range(min(len(self.gold_labels), len(self.predictions))):
                # logger.i('[Evaluation->evaluation] i:{} gold: {}, prediction[i]: {}, \nemotion:{}'.format(i, self.list_gold[i], self.list_prediction[i], self.emotion))
                if self.gold_labels[i] in labels and self.emotion[self.gold_labels[i]] == self.predictions[i]:
                    self.tp += 1
                elif self.predictions[i] == values[labels.index(self.gold_labels[i])] and self.gold_labels[i] not in labels:
                    self.fp += 1
                elif self.gold_labels[i] in self.emotion.keys() and self.predictions[i] != values[labels.index(self.gold_labels[i])]:
                    self.fn += 1
                else:
                    self.tn += 1
            # logger.i('[Evaluation->evaluation] tp: {}, fp: {}, fn:{}, tn:{}'.format(self.tp, self.fp, self.fn, self.tn))
            self.accuracy = (self.tp + self.tn)*1.0 / (self.tp + self.tn + self.fp + self.fn) * 100.
            if (self.tp + self.fp) != 0:
                self.precision = self.tp * 100. / (self.tp + self.fp)
            if (self.tp + self.fn) != 0:
                self.recall = self.tp * 100. / (self.tp + self.fn)
            if self.tp != 0:
                self.F1 = 2. * self.precision * self.recall / (self.precision + self.recall)
            # logger.i('[Evaluation->evaluation] accuracy: {}'.format(self.accuracy))
            logger.i(
                "\n| \tprecision | \trecall | \taccuracy |     \t  F1 |\n"
                "| \t{:9.2f} | \t{:6.2f} | \t{:8.2f} | \t{:8.2f} |".format(
                    self.precision, self.recall, self.accuracy, self.F1))
            # self.print_result(e)

    def print_result(self):
        # logger.i("[Evaluation->print_result]\n")
        # logger.i(e)
        # logger.i(# "| {0:8} |\n"
        #          "| \tprecision | \trecall | \taccuracy |     \t  F1 |\n"
        #          "| \t{1:9.2f} | \t{2:6.2f} | \t{3:8.2f} | \t{4:8.2f} |".format(
        #         self.precision, self.recall, self.accuracy, self.F1))
        pass
