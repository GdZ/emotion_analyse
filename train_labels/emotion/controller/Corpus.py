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
# perception trained labels
from corpus import LABELS_FILE_TXT as LABELS_FILE_TXT
# emb
from corpus import EMB_TRAIN_TXT as EMB_TRAIN_TXT
from corpus import OUTPUT_FILE_MODEL as OUTPUT_FILE_MODEL
# logger
from utils import config
from utils.logger import logger

logger = logger(config)


class Corpus:

    # """
    # init some variables
    # """
    def __init__(self):
        # prepared training text
        self.train_file = PROCESSING_TRAIN_TXT
        # prepared check_accuracy text
        self.test_file = PROCESSING_TEST_TXT
        # marked labels file
        self.label_file = TRAIN_LABEL_CSV
        # gold_labels is created by training labels text
        self.gold_labels = []
        # no use
        self.predicted_labels = []
        # no use
        self.features = []
        # this training corpus is created by prepared training file
        self.train_corpus = []
        # this check_accuracy corpus is created by prepared check_accuracy file
        self.test_corpus = []
        # word list is created by
        self.word_list = []
        self.vector_text = []
        self.tweet_tokens = []

    # """
    # read data from marked label file, and create the gold labels
    # """
    def read_label(self):
        file_handle = io.open_file(self.label_file)
        for line in file_handle:
            self.gold_labels.append(line.strip())
        logger.i('read label successfully')

        return self.gold_labels

    # """
    # create the training corpus data by reading data
    # from the training file. the training corpus will
    # later used in ....
    # """
    def read_train(self):  # different input
        with io.open_file_mode(self.train_file, "r") as r:
            for line in r:
                self.train_corpus.append(line)
        logger.i('read train successfully')

    # """
    # read data from check_accuracy file, to create check_accuracy corpus
    # """
    def read_test(self):  # different input
        with io.open_file_mode(self.test_file, "r") as r:
            for line in r:
                self.test_corpus.append(line)
        logger.i('read check_accuracy successfully')

    # """
    # create the bag of word by using the corpus
    # """
    def bag_of_word(self):
        vector_text = []
        word_list = {}
        last_index = 0
        for text_line in self.train_corpus:
            # every sentence have more than one word,
            # so use a array the save all the vectors
            vector_line = []
            text_split = text_line.strip().split()

            # add word to word list:
            #   word is the index, and index is the value
            for word in text_split:
                if word not in word_list:
                    word_list[word] = last_index
                    last_index += 1

            # save the vector to the vector list
            for token in text_split:
                vector_line.append(word_list[token])

            vector_text.append(vector_line)

        return word_list, vector_text

    # """
    # to create training source data, words and vector list,
    # which are created by words bags
    # """
    def generate_vs_model(self):
        logger.i('generate bag of word...')
        word_list, vector_train = self.bag_of_word()

        logger.i('write %s...' % WORD_LIST_TXT)
        f = io.open_file_mode(WORD_LIST_TXT, "w")
        for (k, v) in word_list.items():
            f.write(str(k))
            f.write(" ")
            f.write(str(v))
            f.write('\n')
        logger.i('write %s finished .......' % WORD_LIST_TXT)

        logger.i('write %s...' % VECTOR_TRAIN_TXT)
        io.write_file(VECTOR_TRAIN_TXT, vector_train)
        logger.i('write %s finished .......' % VECTOR_TRAIN_TXT)

        vector_test = utils.make_test(word_list, self.test_corpus)
        logger.i('write %s...' % VECTOR_TEST_TXT)
        io.write_file(VECTOR_TEST_TXT, vector_test)
        logger.i('write %s finished ......' % VECTOR_TEST_TXT)

        return word_list, vector_train

    # """
    # training perception by use the word list and vector training data
    # """
    def train_perception(self):
        # open 'word_list.txt' file, which have been created by last step
        file_word_list = io.open_file(WORD_LIST_TXT)
        # open 'vector_train.txt' file
        file_vector_train = io.open_file(VECTOR_TRAIN_TXT)
        file_vector_test = io.open_file(VECTOR_TEST_TXT)
        word_list = []
        # store the vector of training data
        vector_train = []
        # store the vector of testing data
        vector_test = []

        logger.d('file_word_list: %s' % type(file_word_list))

        for line in file_word_list:
            # logger.d('line: %s' % line)
            word_list.append(line.strip())
        # logger.d('word_list:\n%s' % word_list)

        logger.i('read word list')
        for line in file_vector_train:
            vector_train.append(line.strip())

        # restore data from file_vector_test to vector_test
        for line in file_vector_test:
            vector_test.append(line.strip())

        logger.i('read vector text for train')
        x, y, w = per.train(word_list, self.gold_labels, vector_train, iteration=20)
        logger.d('[ps->train_perception] x:%d, y:%d, w:%d' % (len(x), len(y), len(w)))

        # create labels for training data
        labels_train = per.generate_labels(x, w)
        # checking correct percent of the training data
        acc = per.check_accuracy(labels_train, y, w)
        logger.i('[ps->train_perception] correct percent of the result: %.4f' % acc)

        # ------------------------------------------------------------------
        # training the label for test
        x_t, y_t, w_t = per.train(word_list, self.gold_labels, vector_test, iteration=20)
        logger.d('[ps->train_perception] x:%d, y:%d, w:%d' % (len(x_t), len(y_t), len(w_t)))

        # create labels for test
        labels_test = per.generate_labels(x_t, w_t)
        logger.d('labels_test(%d) = %d' % (len(labels_test), labels_test[0]))

        # store labels for test to file
        f_l = io.open_file_mode(LABELS_FILE_TXT, "w")
        for l in labels_test:
            # logger.i('l:%d' % l)
            f_l.write(str(l))
            f_l.write('\n')


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

        f = io.open_file_mode(EMB_TRAIN_TXT, 'wb')
        pickle.dump(average, f)

    # training with bayes
    def train_bayes(self):
        naive_bayes(self.train_file, self.label_file, self.test_file)
        # to do: use evaluation
