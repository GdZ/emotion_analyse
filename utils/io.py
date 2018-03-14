# coding: utf-8
import codecs
from utils import config
from utils.logger import logger

logger = logger(config)


def open_file(path):
    logger.i("[io] open file '%s'" % path)
    return open(path)


def read_file(path, mode='r'):
    logger.i("[io] read file '%s' with '%s'" %(path, mode))
    return open(path, mode)


def write_file(path, contents):
    logger.i("[io] write file '%s' with utf-8" % path)
    f = codecs.open(path, "w", "utf-8")
    for line in contents:
        # line = line.encode('utf-8')
        logger.d('line: ' + str(line))

        if isinstance(line[0], int):
            new_line = " ".join(str(x) for x in line)
        else:
            new_line = " ".join(('' + x) for x in line)

        f.write(new_line.replace(u' ä ', u'ä')
                .replace(u' ö ', u'ö')
                .replace(u' ü ', u'ü')
                .replace(u' ß ', u'ß'))
        f.write('\n')


def read_stop_word(stop_word_file):
    stop_word = {}
    logger.i("[io] open file '%s' with read only" % stop_word_file)
    with open(stop_word_file, "r") as r:
        for line in r:
            stop_word[line.strip()] = 1
    return stop_word
