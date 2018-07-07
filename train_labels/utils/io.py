# coding: utf-8
import codecs
import pickle
# self define
from utils import config
from utils.logger import logger

logger = logger(config)


def save(path, variable):
    with open(path, 'w') as f:
        pickle.dump(variable, f)
    f.close()


def load(path):
    with open(path) as f:
        data = pickle.load(f)
    f.close()
    return data


def open_file(path):
    logger.i("[io] open file '%s'" % path)
    return open(path)


def open_file_mode(path, mode='r'):
    logger.i("[io] read file '%s' with '%s'" %(path, mode))
    return open(path, mode)


def write_file(path, contents):
    logger.i("[io->write_file] write file '%s' with utf-8, contents:%d" % (path, len(contents)))
    f = codecs.open(path, "w", "utf-8")
    pos = 1;
    for line in contents:
        logger.d('[io->write_file] [%d]line(%s, %d): %s' %(pos, type(line), len(line), line))
        pos += 1
        if 0 == len(line):
            logger.w('[io->write_file] %s length is: %d' % (line, len(line)))
            f.write('\n')
            continue
        else:
            # line = line.encode('utf-8')
            if isinstance(line[0], int):
                new_line = " ".join(str(x) for x in line)
            else:
                new_line = " ".join(('' + x) for x in line)

            f.write(new_line.replace(u' ä ', u'ä')
                    .replace(u' ö ', u'ö')
                    .replace(u' ü ', u'ü')
                    .replace(u' ß ', u'ß'))
            f.write('\n')
    f.close()


def read_stop_word(stop_word_file):
    stop_word = {}
    logger.i("[io] open file '%s' with read only" % stop_word_file)
    with open(stop_word_file, "r") as r:
        for line in r:
            stop_word[line.strip()] = 1
    return stop_word
