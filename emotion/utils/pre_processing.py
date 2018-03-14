# -*- coding: utf-8 -*-
import re
import string
# logger
from utils import config
from utils.logger import logger

logger = logger(config)


def define_regex():
    # store tokens of tweet content

    # the regular expression of emoji
    emoticons_str = r"""
        (?:
            [:=;] # Eyes
            [oO\-]? # Nose (optional)
            [D\)\]\(\]/\\OpP] # Mouth
        )"""

    # the regular expression of HTML tags, @personName, URLs, numbers and 'NEWLINE'
    # which need to be deleted
    regex_substr = [
        r'<[^>]+>',  # HTML tags
        r'(?:@[\w_]+)',  # @personName
        r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
        r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
        r'NEWLINE'  # the special word 'NEWLINE'
    ]

    # the regular expression of words and anything else which need to be left
    regex_str = [
        emoticons_str,
        r"(?:[a-zäöüß][a-zäöüß'\-_]+[a-zäöüß])",  # words with "-" and "'"
        r'(?:[\w_]+)',  # other words
        r'(?:\S)'  # anything else
    ]

    __tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
    __del_re = re.compile(r'(' + '|'.join(regex_substr) + ')', re.VERBOSE | re.IGNORECASE)
    __hash_re = re.compile(r'(?:\#+)([\w_]+[\w\'_\-]*[\w_]+)')  # Hashtags
    __punc_re = re.compile(r'[%s]' % re.escape(string.punctuation))
    __puncn_re = re.compile(r'[！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀\｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.]',
                            re.VERBOSE | re.IGNORECASE)
    __emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
    return __tokens_re, __del_re, __hash_re, __punc_re, __puncn_re, __emoticon_re


# delete sth. not to be needed & find sth. to be needed
def tokenize(tweet):
    __tokens_re, __del_re, __hash_re, __punc_re, __puncn_re, __emoticon_re = define_regex()
    tweet = __del_re.sub(r'', tweet)  # delete HTML tags, @personName, URLs, numbers and 'NEWLINE'
    # logger.d('2 tweet: ' + tweet.encode('utf-8'))
    tweet = __hash_re.sub(r'\1', tweet)  # delete hash tag but leaving the word after hash tag
    # logger.d('3 tweet: ' + tweet.encode('utf-8'))
    tweet = __puncn_re.sub(r' ', tweet)  # delete chinese punctuation
    # logger.d('4 tweet: ' + tweet.encode('utf-8'))
    tweet = __punc_re.sub(r' ', tweet)  # delete english punctuation
    # logger.d('5 tweet: ' + tweet.encode('utf-8'))
    return __tokens_re.findall(tweet), __emoticon_re


# tokenize
def preprocess(tweet, lowercase=False):
    logger.d('1 tweet: ' + tweet)
    tweet, __emoticon_re = tokenize(tweet)
    if lowercase:  # make capital to lowercase except emoji token
        tweet = [token if __emoticon_re.search(token) else token.lower() for token in tweet]
    # else:
    # tweet = [t.encode('utf-8') for t in tweet ]
    # print('1.1 tweet: ' + str(tweet))
    # pass
    return tweet
