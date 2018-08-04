# coding=utf-8
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

# word cloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# self define
from result import *
from utils import io
# data root folder
from corpus import LABELS_TEST_FILE_TXT as label_test_csv
from corpus import LABELS_TRAIN_FILE_TXT as label_train_csv
from corpus import MARKED_ANALYZE_FILE as marked_csv
from corpus import TRAIN_LABEL_CSV as train_label_csv
from corpus import TEST_LABEL_CSV as test_label_csv
from corpus import STOP_WORD_TXT as stop_word_txt
from corpus import BAYES_TRAIN_LABEL_CSV as bayes_train_csv
from corpus import BAYES_LABELS_TEST_FILE_TXT as bayes_test_csv

# preprocess data
from corpus import PROCESSING_TRAIN_TXT as processing_train_txt
from corpus import BAYES_PROCESSING_TRAIN_TXT as bayes_processing_train_txt
# release
from utils.logger import logger
from utils import config
from emotion.model.Tweet import Item
from emotion.model.Tweet import Tweet
from emotion.model.Labels import Label
from emotion.model.Patient import Patient


class Analyze(Item):
    # const global variables
    tweets = []

    def __init__(self):
        self.logger = logger(config)
        self.logger.i('{} -> __init__'.format(self))
        # *****************************************************************
        # 3.1.*
        # import data from csv
        self.tweets = Tweet(pd.read_csv(marked_csv, sep=';').values)
        self.tweets.keyword = ['feinstaub',
                               # 'feintaubalarm',
                               'vvs','kamine','kaminoefen',
                               'komfort',
                               # 'komfortkamine',
                               # 'moovel'
                               ]
        # self.pdf = self.tweets.get_pdf()
        # self.cdf = np.zeros((np.size(self.tweets.get_x()), self.tweets.SIZE))
        self.x = self.tweets.x
        self.pdf = self.tweets.get_pdf()
        self.cdf = self.tweets.get_cdf()
        # save data to npy
        # np.save('tweets.npy', self.tweets)
        io.save('pkl/tweets.pkl', self.tweets)
        # t = io.load('tweets.pkl')
        # self.logger.i('tweets: {},\nt: {}'.format(self.tweets.pdf[:5], t.pdf[:5]))
        # fig
        self.fig = [] # plt.figure(figsize=(16,9))
        # *****************************************************************
        # labels
        self.label_train = Label(pd.read_csv(label_train_csv).values)
        self.label_test = Label(pd.read_csv(label_test_csv).values)
        self.gold_train = Label(pd.read_csv(train_label_csv).values)
        self.gold_test = Label(pd.read_csv(test_label_csv).values)
        # bayes
        self.bayes_train = Label(pd.read_csv(bayes_train_csv).values)
        self.bayes_test = Label(pd.read_csv(bayes_test_csv).values)
        # texts
        self.process_train_txt = pd.read_csv(processing_train_txt).values
        self.bayes_processing_train_txt = pd.read_csv(bayes_processing_train_txt).values
        self.stop_word_list = pd.read_csv(stop_word_txt).values
        self.alarms = np.asarray([['2016.01.18', '2016.01.22'],['2016.02.26', '2016.02.28'],['2016.03.09', '2016.03.11'],['2016.03.14', '2016.03.22'],['2016.04.10', '2016.04.11'],['2016.10.27', '2016.11.01'],['2016.11.14', '2016.11.15'],['2016.11.21', '2016.12.01'],['2016.12.04', '2016.12.10'],['2016.12.13', '2016.12.23'],['2017.01.09', '2017.01.10'],['2017.01.16', '2017.01.30'],['2017.02.02', '2017.02.03'],['2017.02.07', '2017.02.16'],['2017.02.19', '2017.02.20'],['2017.03.12', '2017.03.17'],['2017.03.25', '2017.04.01'],['2017.04.07', '2017.04.09']])

    def load(self):
        self.logger.i('load begin')
        tweets = io.load('pkl/tweets.pkl')
        x = tweets.x
        ouids = tweets.uids
        uids = np.unique(ouids)
        patients = []
        # 易感，感染，回复
        sus = infec = rec = np.zeros(np.shape(x))
        split = [41, 227, 271, 320, 369, 403]
        for i, uid in enumerate(uids):
            p = io.load('pkl/3.2.1.patients_{}.pkl'.format(uid))
            # max
            for i, s in enumerate(split):
                if 0 != i:
                    idx = np.where(p.pdf[split[i-1]:s] > 0)
                else:
                    idx = np.where(p.pdf[:s] > 0)
                count = np.shape(idx[0])[0]
                # ii = np.where(idx[0] < s)
                if 1 == count:
                    rec[np.min(idx[0][:])] += 1
                if 10 < count:
                    infec[np.min(idx[0][:])] += 2
        self.logger.i('rec: {}'.format(rec))
        self.logger.i('infec: {}'.format(infec))
        ccdf = cdf = np.zeros(shape=np.shape(rec))
        for i, pdf in enumerate(infec):
            cdf[i] = np.sum(infec[:i])
            ccdf[i] = np.sum(rec[:i])
        self.fig = plt.figure(figsize=(16,9))
        # self.fig.add_subplot(2,1,1)
        # plt.plot(x, cdf, 'x-')
        # plt.plot(x, ccdf, '.-')
        # self.fig.add_subplot(2,1,2)
        plt.plot(x, np.log2(rec))
        plt.plot(x, np.log2(ccdf))
        self.fig.savefig('load.png', format='png')
        self.logger.i('load end...')
        # plt.show()
        exit(0)

    def _3_1(self):
        """
        3.1 话题总体时间记大致特点分析
        总体内容：数量，，时间范围， 属性，预处理的结果。🌹
        """
        self.logger.i('_3_1()')
        # print('{}'.format(dates))
        x = self.x
        pdf = self.pdf

        # 日期-数量
        self.fig = plt.figure(figsize=(16,9))
        self.draw(x, pdf[:,self.F_DATE])
        plt.title('date-count of tweets')
        plt.xlabel('Date(days)')
        plt.ylabel('count of tweet everyday')
        self.fig.savefig('3.1.a01.png', format='png')

        # 日期-count(like, retweet, favor)
        self.fig = plt.figure(figsize=(16,21))
        self.fig.add_subplot(3,1,1)
        plt.title('date-count of tweets')
        self.draw(x, pdf[:,F_NR_FAVOR], shape='.', color='b')
        plt.ylabel('count of favor everyday')
        plt.legend(['count of tweets in one day', 'alarm'])
        self.fig.add_subplot(3,1,2)
        self.draw(x, pdf[:,F_NR_REPLY], shape='.', color='g')
        plt.ylabel('count of reply everyday')
        plt.legend(['count of tweets in one day', 'alarm'])
        self.fig.add_subplot(3,1,3)
        self.draw(x, pdf[:,F_NR_RETWEET], shape='.', color='m')
        plt.xlabel('Date(days)')
        plt.ylabel('count of retweet everyday')
        plt.legend(['count of tweets in one day', 'alarm'])
        self.fig.savefig('3.1.a02.png', format='png')

        # 日期-情绪变化
        self.fig = plt.figure(figsize=(16,9))
        self.draw(x, pdf[:, F_LABELSVALUE]/pdf[:, F_DATE])
        plt.title('date-labels of tweets')
        plt.xlabel('Date(days)')
        plt.ylabel('average of labels in everyday')
        self.fig.savefig('3.1.a03.png', format='png')
        plt.show()

    def _3_1_1(self):
        """
        - 日期-不同话题
        :return:
        """
        self.logger.i('_3_1_1')
        texts = self.tweets.texts
        keywords = self.tweets.keyword
        # keywords = ['feinstaub']
        self.fig = plt.figure(figsize=(16,9))
        for i, k in enumerate(keywords):
            x,y = self.keyword_count(k, texts)
            plt.plot(x, y, '.-')
            plt.legend(keywords)
            plt.xticks(x[1:-1:40])
            io.save('3.1.1.{}.x_y--.pkl'.format(i), [x,y])
        self.fig.savefig('3.1.1.a01.png', format='png')
        plt.show()
        self.logger.i('_3_1_1 -> finished')

    def keyword_count(self, keywords, tweets):
        self.logger.i('keyword_count')
        feinstaub = []
        for i, line in enumerate(tweets):
            for k in keywords:
                if k in line.lower():
                    feinstaub.append([self.tweets.dates[i], line])

        feinstaub = np.asarray(feinstaub)
        x = np.unique(feinstaub[:,0])
        y = np.zeros(np.shape(x))
        for i, d in  enumerate(x):
            for j, days in enumerate(feinstaub[:,0]):
                if d == days:
                    y[i] += 1
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
        d = path.dirname(__file__)
        # https://arbeitsschutz-schweissen.de/blog/wp-content/uploads/2015/09/Feinstaub.jpg
        alice_coloring = np.array(Image.open(path.join(d, "mask02.jpg")))

        # 设置停用词
        stopwords = set(STOPWORDS)
        # self.logger.i('stop_word_list: {}'.format(np.shape(self.stop_word_list)))
        # for word in self.stop_word_list:
        #     self.logger.d('word: {}'.format(word))
        #     stopwords.add(str(word))
        stopwords.add('stuttgart')
        stopwords.add('alle')
        stopwords.add('nach')
        stopwords.add('de')
        stopwords.add('fuer')
        stopwords.add('aber')
        stopwords.add('ab')
        stopwords.add('noch')
        stopwords.add('mal')
        stopwords.add('gibt')
        stopwords.add('auch')
        stopwords.add('gilt')
        stopwords.add('nur')
        stopwords.add('ja')
        stopwords.add('nicht')
        stopwords.add('wie')
        stopwords.add('um')
        stopwords.add('swraktuell')
        stopwords.add('gleich')
        stopwords.add('aktuell')
        stopwords.add('index')
        stopwords.add('jetzt')
        stopwords.add('news')
        stopwords.add('stn_news')
        stopwords.add('via')
        stopwords.add('machen')
        stopwords.add('id')
        stopwords.add('swr')

        # 你可以通过 mask 参数 来设置词云形状
        wc = WordCloud(background_color="white",
                       max_words=50,
                       width=640,
                       height=720,
                       # mask=alice_coloring,
                       margin=2,
                       stopwords=stopwords,
                       max_font_size=300,
                       random_state=50)
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
        x = self.tweets.x
        pdf = self.tweets.get_pdf()
        cdf = self.tweets.get_cdf()
        dates = self.tweets.dates
        ppdf = np.zeros(np.shape(x))
        ccdf = np.zeros(np.shape(x))
        self.logger.i('x:{}, y1:{}, dates:{}, pdf:{}'.format(np.shape(x), np.shape(pdf), np.shape(dates), np.shape(ppdf)))
        for i, d in enumerate(x):
            ppdf[i] += np.sum(pdf[i, F_NR_FAVOR:F_NR_RETWEET])
            ccdf[i] += np.sum(ppdf[:i])
        # save
        io.save('3.2.x_pdf_cdf_ppdf_ccdf.pkl', [x, pdf, cdf, ppdf, ccdf])
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
        periods = self.tweets.x
        ouids = self.tweets.uids
        uids = np.unique(ouids)
        dates = self.tweets.dates
        labels = self.tweets.labels
        self.logger.i('uids:\n{}'.format(uids[:5]))
        self.logger.i('periods: {}, uids: {}'.format(periods.shape, uids.shape))
        patients = Patient(uids, periods)
        self.logger.i('patients: -> count:{} staus: {}'.format(patients.patients.shape, patients.patients[0].status.shape))
        # patients.update()
        # for i, d in enumerate(periods):
        for i, uid in enumerate(uids):
            # update patient info on this day
            # ith day of the date
            # 1. use the index to find the day from tweets
            # update all of the patients on this day
            # for j, uid in enumerate(uids):
            for j, d in enumerate(periods):
                # 2. find the index, uid in tweets
                # jth patients:
                # 3. find the user status by uid and day
                # 4. update status
                for k, days in enumerate(dates):
                    stat = nstat = patients.patients[i].status[j]
                    # label = 0
                    if uid == ouids[k] and d == days:
                        nstat = patients.patients[i].update(j)
                    if nstat > 1 and stat < nstat:
                        self.logger.i('i:{}, uid: {}, day: {}\tstatus: {}, nstatus: {}'.format(i, uid, days, stat, nstat))
                    # if 1804 == k:
                    #     self.logger.i('i:{}, uid: {}, day: {}\tstatus: {}, nstatus: {}'.format(i, uid, days, stat, nstat))
            io.save('pkl/3.2.1.patients_{}.pkl'.format(uid), patients.patients[i])
        io.save('pkl/3.2.1.uids.pkl', uids)
        io.save('pkl/3.2.1.periods.pkl', periods)
        io.save('pkl/3.2.1.tweets.pkl', self.tweets)
        self.fig = plt.figure(figsize=(16,9))
        for i, uid in enumerate(uids):
            plt.plot(periods, patients.patients[i].status, '.-')
        self.fig.savefig('3.2.1.a01.png', format='png')
        # plt.show()
        exit(0)

    def _3_2_2(self):
        """
        #
        :return:
        """
        self.logger.i('3.2.2 begin')
        cdf = self.tweets.get_cdf()
        cdf_recover = np.max(cdf) - cdf
        self.logger.i('cdf: {}'.format(cdf.shape))
        io.save('3.2.2.cdf.pkl', cdf)
        # draw
        self.fig = plt.figure(figsize=(16,12))
        self.fig.add_subplot(2,1,1)
        plt.plot(self.x, np.log2(cdf), '.-')
        self.fig.add_subplot(2,1,2)
        plt.plot(self.x, np.log2(cdf_recover), '.-')
        self.fig.savefig('3.2.2.a0.png', format='png')
        # plt.show()
        self.logger.i('3.2.2 finished....')

    def _3_2_3(self):
        """
        #
        :return:
        """
        self.load()
        exit(0)

    def _3_2_4(self):
        """
        #
        :return:
        """
        self.logger.i('无.......')

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
        # train
        mean_train = np.mean(self.label_train.values)
        var_train = np.var(self.label_train.values)
        self.logger.i('mean: {}, var: {}'.format(mean_train, var_train))
        self.fig = plt.figure(figsize=(16,18))
        self.fig.add_subplot(2,2,1)
        n1, bins1, patches1 = plt.hist(self.gold_train.values)
        plt.ylabel('predict of gold labels set')
        mu, sigma = np.mean(self.gold_train.values), np.var(self.gold_train.values)
        y1 = mlab.normpdf(bins1, mu, sigma) + mu
        self.logger.i('bins:{}, y:{}'.format(bins1, y1))
        self.fig.add_subplot(2,2,2)
        plt.plot(bins1, y1, '.-')
        self.fig.add_subplot(2,2,3)
        n2, bins2, patches2 = plt.hist(self.label_train.values)
        plt.ylabel('predict of train set')
        self.fig.add_subplot(2,2,4)
        y2 = mlab.normpdf(bins2, mean_train, var_train)
        plt.plot(bins2, y2, '.-')
        self.fig.savefig('3.3.1.a01.png', format='png')
        # io.save('3.3.1.train.bins_bins_y.pkl')

        # test
        mean_test = np.mean(self.label_test.values)
        var_test = np.var(self.label_test.values)
        self.logger.i('mean: {}, var: {}'.format(mean_test, var_test))
        self.fig = plt.figure(figsize=(16,18))
        self.fig.add_subplot(2,2,1)
        n, bins, patches = plt.hist(self.label_test.values)
        plt.ylabel('predict of test set')
        self.fig.add_subplot(2,2,2)
        y = mlab.normpdf(bins, np.mean(self.label_test.values), np.var(self.label_test.values))
        plt.plot(bins, y, '.-')
        self.fig.add_subplot(2,2,3)
        n, bins, patches = plt.hist(self.gold_test.values)
        plt.ylabel('gold label of test set')
        self.fig.add_subplot(2,2,4)
        y = mlab.normpdf(bins, mean_test, var_test)
        plt.plot(bins, y, '.-')
        self.fig.savefig('3.3.1.a02.png', format='png')
        # plt.show()
        # exit(0)

    def _3_3_2(self):
        """
        # bayes
        :return:
        """
        # train
        mean_train = np.mean(self.bayes_train.values)
        var_train = np.var(self.bayes_train.values)
        self.logger.i('mean: {}, var: {}'.format(mean_train, var_train))
        self.fig = plt.figure(figsize=(16,18))
        self.fig.add_subplot(2,2,1)
        n, bins, patches = plt.hist(self.bayes_train.values)
        plt.ylabel('predict of gold labels set')
        mu, sigma = np.mean(self.bayes_train.values), np.var(self.bayes_train.values)
        y = mlab.normpdf(bins, mu, sigma) + mu
        self.logger.i('bins:{}, y:{}'.format(bins, y))
        self.fig.add_subplot(2,2,2)
        plt.plot(bins, y, '.-')
        self.fig.add_subplot(2,2,3)
        n, bins, patches = plt.hist(self.label_train.values)
        plt.ylabel('predict of train set')
        self.fig.add_subplot(2,2,4)
        y = mlab.normpdf(bins, mean_train, var_train)
        plt.plot(bins, y, '.-')
        self.fig.savefig('3.3.1.a01.png', format='png')

        # test
        mean_test = np.mean(self.label_test.values)
        var_test = np.var(self.label_test.values)
        # cov_test = np.cov(self.label_test, self.gold_test)
        self.logger.i('mean: {}, var: {}'.format(mean_test, var_test))
        self.fig = plt.figure(figsize=(16,18))
        # plt.scatter(self.label_test, self.gold_test)
        # plt.plot(self.gold_train, self.label_train, 'r.-')
        # plt.plot(self.gold_test, self.label_test, 'b.')
        self.fig.add_subplot(2,2,1)
        n, bins, patches = plt.hist(self.label_test.values)
        plt.ylabel('predict of test set')
        self.fig.add_subplot(2,2,2)
        y = mlab.normpdf(bins, np.mean(self.label_test.values), np.var(self.label_test.values))
        plt.plot(bins, y, '.-')
        self.fig.add_subplot(2,2,3)
        n, bins, patches = plt.hist(self.gold_test.values)
        plt.ylabel('gold label of test set')
        self.fig.add_subplot(2,2,4)
        y = mlab.normpdf(bins, mean_test, var_test)
        plt.plot(bins, y, '.-')
        self.fig.savefig('3.3.1.a02.png', format='png')
        # plt.show()
        exit(0)

    def _3_3_3(self):
        """
        #
        :return:
        """
        patients = io.load('3.2.1.patients.pkl')
        for i, p in enumerate(patients.patients):
            self.fig = plt.figure(figsize=(16,9))
            plt.plot(self.x, p.status, '.-')

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

    # def draw(self, x, y, shape='r.-', xlabel='', ylabel='', title='', lengend=''):
    def draw(self, x, y, shape='.', color='b'):
        self.logger.i('draw')
        idx = []
        color = ['r{}'.format(shape), '{}-{}'.format(color, shape)]

        plt.plot(x, y, color[1])

        for al in self.alarms:
            tmp = [np.where(x == al[0])[0][0], np.where(x == al[1])[0][0]]
            idx.append(tmp)
        idx = np.asarray(idx)

        for xx in idx:
            pos = xx
            xxx = np.linspace(pos[0], pos[1], num=pos[1]-pos[0]+1, dtype=int)
            plt.plot(xxx, y[xxx], color[0])
        plt.xticks(x[1:-1:40])

        self.logger.i('draw -> finished.')

    def show(self):
        plt.show()

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
                self.show()
                break
            else:
                self.load()
                break
        self.logger.i('task -> finished')
