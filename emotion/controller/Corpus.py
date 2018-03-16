# coding: utf-8
"""
this file contain some method to deal with different framework to create corpus library
"""
import gensim
import pickle
# utils
from utils import io
# emotion package
from emotion import utils
from emotion.model import Perception as per
from emotion.model.Bayes import naive_bayes
import emotion.model.Embedding
# const variable
from corpus import WORD_LIST_TXT
from corpus import PROCESSING_TRAIN_TXT as PROCESSING_TRAIN_TXT
from corpus import PROCESSING_TEST_TXT as PROCESSING_TEST_TXT
from corpus import TRAIN_LABEL_CSV as TRAIN_LABEL_CSV
from corpus import VECTOR_TRAIN_TXT as VECTOR_TRAIN_TXT
from corpus import VECTOR_TEST_TXT as VECTOR_TEST_TXT
from corpus import EMB_TRAIN_TXT as EMB_TRAIN_TXT
from corpus import OUTPUT_FILE_MODEL as OUTPUT_FILE_MODEL
# logger
from utils import config
from utils.logger import logger

logger = logger(config)


class Corpus:

    def __init__(self):
        self.train_file = PROCESSING_TRAIN_TXT
        self.test_file = PROCESSING_TEST_TXT
        self.label_file = TRAIN_LABEL_CSV
        self.gold_labels = []
        self.predicted_labels = []
        self.features = []
        self.train_corpus = []
        self.test_corpus = []
        self.word_list = []
        self.vector_text = []
        self.tweet_tokens = []

    # read label data
    def read_label(self):
        file_handle = io.open_file(self.label_file)
        for line in file_handle:
            self.gold_labels.append(line.strip())
        logger.i('read label successfully')

        return self.gold_labels

    # read training file
    def read_train(self):  # different input
        with io.read_file(self.train_file, "r") as r:
            for line in r:
                self.train_corpus.append(line)
        logger.i('read train successfully')

    # read test data
    def read_test(self):  # different input
        with io.read_file(self.test_file, "r") as r:
            for line in r:
                self.test_corpus.append(line)
        logger.i('read test successfully')

    #
    def bag_of_word(self):
        vector_text = []
        word_list = {}
        last_index = 0
        for text_line in self.train_corpus:
            vector_line = []
            text_split = text_line.strip().split()

            # add word to word_list
            for word in text_split:
                if word not in word_list:
                    word_list[word] = last_index
                    last_index += 1

            for token in text_split:
                vector_line.append(word_list[token])
            vector_text.append(vector_line)

        return word_list, vector_text

    # generate a model
    def generate_vs_model(self):
        logger.i('generate bag of word...')
        word_list, vector_train = self.bag_of_word()
        logger.i('write word_list.txt...')
        f = io.read_file(WORD_LIST_TXT, "w")

        for (k, v) in word_list.items():
            f.write(str(k))
            f.write(" ")
            f.write(str(v))
            f.write('\n')

        logger.i('write vector_train.txt...')
        io.write_file(VECTOR_TRAIN_TXT, vector_train)

        vector_test = utils.make_test(word_list, self.test_corpus)
        logger.i('write vector_test.txt...')
        io.write_file(VECTOR_TEST_TXT, vector_test)

        return word_list, vector_train

    # training with perception
    def train_perception(self):
        file_word_list = io.open_file(WORD_LIST_TXT)
        file_vector_train = io.open_file(VECTOR_TRAIN_TXT)
        word_list = []
        vector_train = []

        logger.d('file_word_list: %s' % type(file_word_list))

        for line in file_word_list:
            # logger.d('line: %s' % line)
            word_list.append(line.strip())
        # logger.d('word_list:\n%s' % word_list)

        logger.i('read word list')
        for line in file_vector_train:
            vector_train.append(line.strip())

        logger.i('read vector text for train')
        x, y, w = per.train(word_list, self.gold_labels, vector_train, iteration=20)
        logger.d('[ps->train_perception] x:%d, y:%d, w:%d' % (len(x), len(y), len(w)))

        acc = per.test(x, y, w)
        logger.i('[ps->train_perception] correct percent of the result: %.4f' % acc)

    # use embedding
    @staticmethod
    def embedding():
        vector_train = []
        file_vector_train = io.open_file(VECTOR_TRAIN_TXT)

        for line in file_vector_train:
            vector_train.append(line.strip())

        emotion.model.Embedding.generate_model(vector_train)
        model = gensim.models.Word2Vec.load(OUTPUT_FILE_MODEL)
        average = emotion.model.Embedding.make_average(vector_train[0:10], model)

        f = io.read_file(EMB_TRAIN_TXT, 'wb')
        pickle.dump(average, f)

    # training with bayes
    def train_bayes(self):
        naive_bayes(self.train_file, self.label_file, self.test_file)
        # to do: use evaluation
