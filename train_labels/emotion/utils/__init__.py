# coding: utf-8
import codecs
from utils import config
from utils.logger import logger

logger = logger(config)


def make_test(word_list, text):
    vector_text = []
    pos = 1
    for line in text:
        vector_line = []
        line = line.split()
        for word in line:
            if word in word_list:
                index = word_list[word]
                vector_line.append(index)
        vector_text.append(vector_line)
        pos += 1
    return vector_text
