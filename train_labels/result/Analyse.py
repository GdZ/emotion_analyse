# coding=utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats
# word cloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# self define
from result import *
# data root folder
from corpus import LABELS_TEST_FILE_TXT as label_test_csv
from corpus import LABELS_TRAIN_FILE_TXT as label_train_csv
from corpus import MARKED_ANALYZE_FILE as marked_csv
from corpus import TRAIN_LABEL_CSV as train_label_csv
from corpus import TEST_LABEL_CSV as test_label_csv
from corpus import STOP_WORD_TXT as stop_word_txt
# preprocess data
from corpus import PROCESSING_TRAIN_TXT as processing_train_txt
from corpus import BAYES_PROCESSING_TRAIN_TXT as bayes_processing_train_txt
# release
from utils.logger import logger
from utils import config

class Analyze():
    # const global variables
    F_ID = 0                                # ID
    F_USER_ID = F_ID + 1                    # user_id
    F_TYPE = F_USER_ID + 1                  # type
    F_SENTIMENTLABEL = F_TYPE + 1           # Sentiment label
    F_LABELSVALUE = F_SENTIMENTLABEL + 1    # labels value
    F_USERNAMETWEET = F_LABELSVALUE + 1     # usernameTweet
    F_TEXT = F_USERNAMETWEET + 1            # text
    F_IS_REPLY = F_TEXT + 1                 # is_reply
    F_IS_RETWEET = F_IS_REPLY + 1           # is_retweet
    F_NR_FAVOR = F_IS_RETWEET + 1           # nbr_favorite
    F_NR_REPLY = F_NR_FAVOR + 1             # nbr_reply
    F_NR_RETWEET = F_NR_REPLY + 1           # nbr_retweet
    F_DATE = F_NR_RETWEET + 1               # date
    F_TIME = F_DATE + 1                     # time
    F_HAS_MEDIA = F_TIME + 1                # has_media
    F_MEDIAS0 = F_HAS_MEDIA + 1             # medias.0
    F_URL = F_MEDIAS0 + 1                   # url

    def __init__(self):
        self.logger = logger(config)
        self.logger.i('__init__')
        # *****************************************************************
        # 3.1.*
        # import data from csv
        fd = pd.read_csv(marked_csv, sep=';')
        # format
        self.data = fd.values
        self.ids = self.data[:, F_ID]
        self.uids = self.data[:, F_USER_ID]
        self.types = self.data[:, F_TYPE]
        self.sentiments = self.data[:, F_SENTIMENTLABEL]
        self.labels = self.data[:, F_LABELSVALUE]
        self.unames = self.data[:, F_USERNAMETWEET]
        self.texts = self.data[:, F_TEXT]
        self.is_reply = self.data[:, F_IS_REPLY]
        self.is_retweet = self.data[:, F_IS_RETWEET]
        self.nr_favor = self.data[:, F_NR_FAVOR]
        self.nr_reply = self.data[:, F_NR_REPLY]
        self.nr_retweet = self.data[:, F_NR_RETWEET]
        self.dates = self.data[:, F_DATE]
        self.times = self.data[:, F_TIME]
        self.medias = self.data[:, F_HAS_MEDIA]
        self.media0 = self.data[:, F_MEDIAS0]
        self.urls = self.data[:, F_URL]
        self.fig = []; # plt.figure(figsize=(16,9))
        # *****************************************************************
        # 3.3.*
        # self.label_train = pd.read_csv(label_train_csv, sep=';').values
        # self.label_test = pd.read_csv(label_test_csv, sep=';').values
        # self.gold_train = pd.read_csv(train_label_csv, sep=';').values
        # self.gold_test = pd.read_csv(test_label_csv, sep=';').values
        self.label_train = pd.read_csv(label_train_csv).values
        self.label_test = pd.read_csv(label_test_csv).values
        self.gold_train = pd.read_csv(train_label_csv).values
        self.gold_test = pd.read_csv(test_label_csv).values
        self.process_train_txt = pd.read_csv(processing_train_txt).values
        self.bayes_processing_train_txt = pd.read_csv(bayes_processing_train_txt).values
        self.stop_word_list = pd.read_csv(stop_word_txt).values
        self.xDate = np.unique(self.dates)
        # print(xDate)
        # count
        self.pdf = np.zeros((np.size(self.xDate), 17))
        self.cdf = np.zeros((np.size(self.xDate), 17))
        print('y1: {}'.format(self.pdf.shape))
        for i, d in enumerate(self.xDate):
            for j, days in enumerate(self.dates):
                if d == days:
                    self.pdf[i][self.F_DATE] += 1
                    self.pdf[i][self.F_NR_FAVOR] += self.nr_favor[j]
                    self.pdf[i][self.F_NR_REPLY] += self.nr_reply[j]
                    self.pdf[i][self.F_NR_RETWEET] += self.nr_retweet[j]
                    self.pdf[i][self.F_LABELSVALUE] += self.labels[j]

    def _3_1(self):
        """
        3.1 话题总体时间记大致特点分析
        总体内容：数量，，时间范围， 属性，预处理的结果。🌹
        """
        self.logger.i('_3_1()')
        # print('{}'.format(dates))
        x = self.xDate
        pdf = self.pdf

        # 日期-数量
        self.fig = plt.figure(figsize=(16,9))
        self.draw(x, pdf[:,self.F_DATE], 'r-x', title='date-count of tweets',
                  xlabel='Date(days)', ylabel='count of tweet everyday')
        self.fig.savefig('3.1.a01.png', format='png')

        # 日期-count(like, retweet, favor)
        self.fig = plt.figure(figsize=(16,9))
        self.draw(x, pdf[:,F_NR_FAVOR], shape='r.-')
        self.draw(x, pdf[:,F_NR_REPLY], shape='b.-')
        self.draw(x, pdf[:,F_NR_RETWEET], shape='g.-', title='date-count of tweets',
                  xlabel='Date(days)', ylabel='count of tweet everyday',
                  lengend=['count of favor from all tweets in one day',
                           'count of reply from all tweets in one day',
                           'count of retweet from all tweets in one day'])
        self.fig.savefig('3.1.a02.png', format='png')

        # 日期-情绪变化
        self.fig = plt.figure(figsize=(16,9))
        self.draw(x, pdf[:, F_LABELSVALUE]/pdf[:, F_DATE], shape='r-*',
                  title='date-labels of tweets',
                  xlabel='Date(days)', ylabel='average of labels in everyday')
        self.fig.savefig('3.1.a03.png', format='png')
        plt.show()

    def _3_1_1(self):
        """
        - 日期-不同话题
            - feinstaub
            - feintaubalarm
            - vvs
            - kamine
            - kaminoefen
            - komfort
            - komfortkamine
            - moovel
        :return:
        """
        self.logger.i('_3_1_1')
        texts = self.texts
        self.fig = plt.figure(figsize=(16,9))
        x,y = self.keyword_count('feinstaub', texts)
        plt.plot(x, y, '.-')
        x,y = self.keyword_count('feinstaubalarm', texts)
        # fig = plt.figure(figsize=(16,9))
        plt.plot(x, y, '.-')
        x,y = self.keyword_count('vvs', texts)
        # fig = plt.figure(figsize=(16,9))
        plt.plot(x, y, '.-')
        x,y = self.keyword_count('kamine', texts)
        # fig = plt.figure(figsize=(16,9))
        plt.plot(x, y, '.-')
        x,y = self.keyword_count('kaminoefen', texts)
        # fig = plt.figure(figsize=(16,9))
        plt.plot(x, y, '.-')
        x,y = self.keyword_count('komfort', texts)
        # fig = plt.figure(figsize=(16,9))
        plt.plot(x, y, '.-')
        x,y = self.keyword_count('komfortkamine', texts)
        # fig = plt.figure(figsize=(16,9))
        plt.plot(x, y, '.-')
        x,y = self.keyword_count('moovel', texts)
        # fig = plt.figure(figsize=(16,9))
        plt.plot(x, y, '.-')
        plt.legend(['feinstaub','feintaubalarm','vvs','kamine','kaminoefen','komfort','komfortkamine','moovel'])
        self.fig.savefig('3.1.1.a01.png', format='png')
        plt.show()
        self.logger.i('_3_1_1 -> finished')

    def keyword_count(self, keywords, tweets):
        self.logger.i('keyword_count')
        feinstaub = []
        for i, line in enumerate(tweets):
            for k in keywords:
                if k in line.lower():
                    feinstaub.append([self.dates[i], line]);

        feinstaub = np.asarray(feinstaub)
        x = np.unique(feinstaub[:,0])
        y = np.zeros(np.shape(x))
        for i, d in  enumerate(x):
            for j, days in enumerate(feinstaub[:,0]):
                if d == days:
                    y[i] += 1;
        self.logger.i('keyword_count -> finished')
        return x,y

    def _3_1_2(self):
        """
        对总体的分析总结，包括整体的唔买预警机制下关注的水平及背后的原理说明
        :return:
        """
        pass

    def _3_1_3(self):
        """
        推文显著关键词联想图标
        :return:
        """
        # from wordcloud import WordCloud
        # self.logger.i()
        # return ;
        # wordcloud = WordCloud(
        #     background_color="white",
        #     width=1000,
        #     height=860,
        #     margin=2
        # ).generate(str(list(self.bayes_processing_train_txt)))
        # # width,height,margin可以设置图片属性
        # # generate 可以对全部文本进行自动分词,但是他对中文支持不好,对中文的分词处理请看我的下一篇文章
        # # wordcloud = WordCloud(font_path = r'D:\Fonts\simkai.ttf').generate(f)
        # # 你可以通过font_path参数来设置字体集
        # # background_color参数为设置背景颜色,默认颜色为黑色
        # plt.imshow(wordcloud)
        # plt.axis("off")
        # plt.show()
        # 保存图片,但是在第三模块的例子中 图片大小将会按照 mask 保存
        """
        Image-colored wordcloud
        =======================
        
        You can color a word-cloud by using an image-based coloring strategy
        implemented in ImageColorGenerator. It uses the average color of the region
        occupied by the word in a source image. You can combine this with masking -
        pure-white will be interpreted as 'don't occupy' by the WordCloud object when
        passed as mask.
        If you want white as a legal color, you can just pass a different image to
        "mask", but make sure the image shapes line up.
        """
        d = path.dirname(__file__)
        # https://arbeitsschutz-schweissen.de/blog/wp-content/uploads/2015/09/Feinstaub.jpg
        alice_coloring = np.array(Image.open(path.join(d, "mask02.jpg")))

        # 设置停用词
        stopwords = set(STOPWORDS)
        self.logger.i('stop_word_list: {}'.format(np.shape(self.stop_word_list)))
        for word in self.stop_word_list:
            self.logger.d('word: {}'.format(word))
            stopwords.add(str(word))

        # 你可以通过 mask 参数 来设置词云形状
        wc = WordCloud(background_color="white",
                       max_words=2000,
                       width=640,
                       height=720,
                       # mask=alice_coloring,
                       margin=2,
                       stopwords=stopwords,
                       max_font_size=200,
                       random_state=42)
        # generate word cloud
        texts = ''
        for w in self.process_train_txt:
            txt = w[0]
            self.logger.d('bevor word: {}'.format(txt))
            txt = txt.replace('ö', 'oe')
            txt = txt.replace('ä', 'ae')
            txt = txt.replace('ü', 'ue')
            txt = txt.replace('Ö', 'Oe')
            txt = txt.replace('Ä', 'Ae')
            txt = txt.replace('Ü', 'Ue')
            txt = txt.replace('ß', 'ss')
            self.logger.d('nach word: {}'.format(txt))
            texts += txt
        wc.generate(str(texts))

        # show
        # 在只设置mask的情况下,你将会得到一个拥有图片形状的词云
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        wc.to_file('3.1.3.a01.png')
        plt.show()

    def _3_2(self):
        """
        # 推文信息扩散模型模拟
        每日推文：favor + reply + retweet
        :return:
        """
        x = self.xDate
        pdf = self.pdf
        dates = self.dates
        ppdf = np.zeros(np.shape(x))
        ccdf = np.zeros(np.shape(x))
        self.logger.i('x:{}, y1:{}, dates:{}, pdf:{}'.format(np.shape(x), np.shape(pdf), np.shape(dates), np.shape(ppdf)))
        for i, d in enumerate(x):
            ppdf[i] += np.sum(pdf[i, F_NR_FAVOR:F_NR_RETWEET])
            ccdf[i] += np.sum(ppdf[:i])
        # draw
        self.fig = plt.figure(figsize=(16,12))
        self.fig.add_subplot(2,2,1)
        plt.plot(x, ppdf)
        plt.title('pdf of spread tweets')
        self.fig.add_subplot(2,2,2)
        plt.plot(x, np.log(ppdf))
        plt.title('log(pdf) of spread tweets')
        self.fig.add_subplot(2,2,3)
        plt.plot(x, ccdf)
        plt.title('cdf of spread tweets')
        self.fig.add_subplot(2,2,4)
        plt.plot(x, np.log(ccdf))
        plt.title('log(cdf) of spread tweets')
        self.fig.savefig('3.2.a0.png', format='png')
        # plt.show()

    def _3_2_1(self):
        """
        # 感染，痊愈，死亡
        :return:
        """
        pass

    def _3_2_2(self):
        """
        #
        :return:
        """
        pass

    def _3_2_3(self):
        """
        #
        :return:
        """
        pass

    def _3_2_4(self):
        """
        #
        :return:
        """
        pass

    def _3_3(self):
        """
        #
        :return:
        """
        pass

    def _3_3_1(self):
        """
        感知器分析结果
        :return:
        """
        self.logger.i('label train: {}'.format(np.shape(self.label_train)))
        self.logger.i('label test: {}'.format(np.shape(self.label_test)))
        self.logger.i('gold train: {}'.format(np.shape(self.gold_train)))
        self.logger.i('gold test: {}'.format(np.shape(self.gold_test)))

        # train
        mean_train = np.mean(self.label_train)
        var_train = np.var(self.label_train)
        # cov_train = np.cov(self.label_train, self.gold_train)
        # self.logger.i('cov: {}'.format(cov_train))
        self.logger.i('mean: {}, var: {}'.format(mean_train, var_train))
        self.fig = plt.figure(figsize=(16,18))
        # plt.scatter(self.label_train, self.gold_train)
        self.fig.add_subplot(2,1,1)
        plt.hist(self.gold_train)
        self.fig.add_subplot(2,1,2)
        plt.hist(self.label_train)
        # plt.plot(self.gold_train, self.label_train, 'r.-')
        # self.fig.show()
        self.fig.savefig('3.3.1.a01.png', format='png')

        # test
        mean_test = np.mean(self.label_test)
        var_test = np.var(self.label_test)
        # cov_test = np.cov(self.label_test, self.gold_test)
        self.logger.i('mean: {}, var: {}'.format(mean_test, var_test))
        self.fig = plt.figure(figsize=(16,18))
        # plt.scatter(self.label_test, self.gold_test)
        # plt.plot(self.gold_train, self.label_train, 'r.-')
        # plt.plot(self.gold_test, self.label_test, 'b.')
        self.fig.add_subplot(2,1,1)
        plt.hist(self.label_test)
        plt.ylabel('predict of test set')
        self.fig.add_subplot(2,1,2)
        plt.hist(self.gold_test)
        plt.ylabel('gold label of test set')
        # self.fig.show()
        plt.show()
        self.fig.savefig('3.3.1.a02.png', format='png')

    def _3_3_2(self):
        """
        #
        :return:
        """
        pass

    def _3_3_3(self):
        """
        #
        :return:
        """
        pass

    def _3_3_4(self):
        """
        #
        :return:
        """
        pass

    def _3_3_5(self):
        """
        #
        :return:
        """
        pass

    def draw(self, x, y, shape='r.-', xlabel='', ylabel='', title='', lengend=''):
        self.logger.i('draw')
        plt.plot(x, y, shape);
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(lengend)
        # plt.grid(True)
        self.logger.i('draw -> finished.')

    def save(self):
        self.logger.i('save')
        for name in self.figlist:
            self.fig.savefig(name, format='png')
        self.logger.i('save finished')

    def task(self):
        self.logger.i('task')
        menu = ("choose your task:\n"
                "-----------------\n"
                "0. exit this part\n"
                "1. 3.1.\n"
                "2. 3.1.1.\n"
                "3. 3.1.2.\n"
                "4. 3.1.3.\n"
                "5. 3.2.\n"
                "6. 3.2.1.\n"
                "7. 3.2.2.\n"
                "8. 3.2.3.\n"
                "9. 3.2.4.\n"
                "10. 3.3.\n"
                "11. 3.3.1.\n"
                "12. 3.3.2.\n"
                "13. 3.3.3.\n"
                "14. 3.3.4.\n"
                "15. 3.3.5.\n"
                "-----------------\n")
        while True:
            op = input(menu)
            if 1 == op:
                self._3_1()
            elif 2 == op:
                self._3_1_1()
            elif 3 == op:
                self._3_1_2()
            elif 4 == op:
                self._3_1_3()
            elif 5 == op:
                self._3_2()
            elif 6 == op:
                self._3_2_1()
            elif 7 == op:
                self._3_2_2()
            elif 8 == op:
                self._3_2_3()
            elif 9 == op:
                self._3_2_4()
            elif 10 == op:
                self._3_3()
            elif 11 == op:
                self._3_3_1()
            elif 12 == op:
                self._3_3_2()
            elif 13 == op:
                self._3_3_3()
            elif 14 == op:
                self._3_3_4()
            elif 15 == op:
                self._3_3_5()
            elif 0 == op:
                break
        self.logger.i('task -> finished')
