# coding: utf-8
# self define
from utils import io
from emotion.utils import pre_processing as pp
from emotion.controller.Corpus import Corpus
# const variable
from corpus import TRAIN_CSV as TRAIN_CSV
from corpus import TEST_CSV as TEST_CSV
from corpus import STOP_WORD_TXT as STOP_WORD_TXT
from corpus import PROCESSING_TRAIN_TXT as PROCESSING_TRAIN_TXT
from corpus import PROCESSING_TEST_TXT as PROCESSING_TEST_TXT
# idx
from corpus import TRAIN_CONTEXT_IDX as TRAIN_CONTEXT_IDX
from corpus import SPLIT_STR as SPLIT_STR
# release
from utils.logger import logger
from utils import config

logger = logger(config)


def prepare(ch=1):
    contents = []
    pre_processed = []
    stop_words = io.read_stop_word(STOP_WORD_TXT)

    if 1 == ch:
        fd = io.open_file(TRAIN_CSV)

    elif 2 == ch:
        fd = io.open_file(TEST_CSV)

    for line in fd:
        try:
            line_split = line.decode('utf-8').strip().split(SPLIT_STR, TRAIN_CONTEXT_IDX)
            logger.d('line_split: ' + line_split[TRAIN_CONTEXT_IDX])
            new_line = line_split[TRAIN_CONTEXT_IDX]
            # contents.append(new_line)
            contents.append(new_line.lower())
        except IndexError, e:
            logger.e(IndexError, e)

    logger.i('delete stop words')
    for twitter in contents:
        logger.d('twitter: ' + twitter)
        tweet_tokens = pp.preprocess(twitter)
        tweet_tokens_final = []
        for token in tweet_tokens:
            if token in stop_words:
                pass
            else:
                logger.d('token: ' + token)
                tweet_tokens_final.append(token)
        pre_processed.append(tweet_tokens_final)

    if 1 == ch:
        io.write_file(PROCESSING_TRAIN_TXT, pre_processed)
    elif 2 == ch:
        io.write_file(PROCESSING_TEST_TXT, pre_processed)
    logger.i('pre processing is finished.....')


def training(option='perception'):
    train_corpus = Corpus()
    train_corpus.read_train()
    train_corpus.read_label()
    train_corpus.read_test()

    # perception-> use bag of words and perception
    # [3] create vector
    if 'vector' == option:
        logger.i("create by vector....")
        train_corpus.generate_vs_model()

    # [4] perception
    elif 'perception' == option:
        logger.i("training by perception....")
        train_corpus.train_perception()

    # [5] bayes
    elif 'bayes' == option:
        logger.i("training by use bayes")
        train_corpus.train_bayes()

    # [6] emb
    elif 'emb' == option:
        logger.i("training by use emb....")
        train_corpus.embedding()

    else:
        logger.i("do nothing.....")

    logger.i('training is finished.....')


def main():
    menu = ("\n"
            "---------------------------------\n"
            "1. pre-process train\n"
            "2. pre-process check_accuracy\n"
            "3. training with vector\n"
            "4. training with perception\n"
            "5. training with bayes\n"
            "6. training with emb\n"
            "0. exit\n"
            "---------------------------------\n")
    ch = input(menu)

    if 1 == ch:
        prepare(ch)
    elif 2 == ch:
        prepare(ch)
    elif 3 == ch:
        option = 'vector'
        training(option)
    elif 4 == ch:
        option = 'perception'
        training(option)
    elif 5 == ch:
        option = 'bayes'
        training(option)
    elif 6 == ch:
        option = 'emb'
        training(option)
    elif 0 == ch:
        exit(0)


if __name__ == "__main__":
    while True:
        main()
        logger.i("finished....")
