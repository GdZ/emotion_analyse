# coding: utf-8
import gensim
import logging
import os
# local
from corpus import OUTPUT_FILE_MODEL as OUTPUT_FILE_MODEL
# debug
from utils import config
from utils.logger import logger


logger = logger(config)


# using average to save only one vector for a sentence
def make_average(sentences, model):
    average = []
    index = 0
    for sentence in sentences:
        try:
            length = len(model[sentence[0]])
            average_temp = []
            for i in range(length):
                temp = 0
                for word in sentence:
                    temp += model[word][i]
                temp /= len(sentence)
                average_temp.append(temp)
            average.append(average_temp)
        except IndexError, e:
            logger.i('[Embedding->make_average] index: %d' % index)
            logger.e(IndexError, e)
        index += 1
    return average


def generate_model(vector_text):
    model = gensim.models.Word2Vec(vector_text, min_count=1, size=100, workers=4)
    logger.i('[Embedding->generate_model] save to %s' % OUTPUT_FILE_MODEL)
    model.save(OUTPUT_FILE_MODEL)
