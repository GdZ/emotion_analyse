import numpy as np


class Item:
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
    SIZE = F_URL + 1


class Tweet(Item):
    def __init__(self, data):
        self.dataset = data
        self.shape = np.shape(self.dataset)
        self.ids = self.dataset[:, self.F_ID]
        self.uids = self.dataset[:, self.F_USER_ID]
        self.types = self.dataset[:, self.F_TYPE]
        self.sentiments = self.dataset[:, self.F_SENTIMENTLABEL]
        self.labels = self.dataset[:, self.F_LABELSVALUE]
        self.unames = self.dataset[:, self.F_USERNAMETWEET]
        self.texts = self.dataset[:, self.F_TEXT]
        self.is_reply = self.dataset[:, self.F_IS_REPLY]
        self.is_retweet = self.dataset[:, self.F_IS_RETWEET]
        self.nr_favor = self.dataset[:, self.F_NR_FAVOR]
        self.nr_reply = self.dataset[:, self.F_NR_REPLY]
        self.nr_retweet = self.dataset[:, self.F_NR_RETWEET]
        self.dates = self.dataset[:, self.F_DATE]
        self.times = self.dataset[:, self.F_TIME]
        self.medias = self.dataset[:, self.F_HAS_MEDIA]
        self.media0 = self.dataset[:, self.F_MEDIAS0]
        self.urls = self.dataset[:, self.F_URL]
        self.keyword = []
        # new define
        self.x = np.unique(self.dates)
        self.pdf = np.zeros((np.size(self.x), self.SIZE))
        self.cdf = np.zeros((np.size(self.x), self.SIZE))

    # def get_dataset(self):
    #     return self.dataset
    #
    # def get_x(self):
    #     return self.x

    def get_pdf(self):
        print('y1: {}'.format(self.pdf.shape))
        for i, d in enumerate(self.x):
            for j, days in enumerate(self.dates):
                if d == days:
                    self.pdf[i, self.F_ID] += 1
                    self.pdf[i, self.F_DATE] += 1
                    self.pdf[i, self.F_NR_FAVOR] += self.nr_favor[j]
                    self.pdf[i, self.F_NR_REPLY] += self.nr_reply[j]
                    self.pdf[i, self.F_NR_RETWEET] += self.nr_retweet[j]
                    self.pdf[i, self.F_LABELSVALUE] += self.labels[j]
                    uids = []
                    if self.uids[j] not in uids:
                        uids.append(self.uids[j])
                        self.pdf[i, self.F_USER_ID] += 1
        return self.pdf

    def get_cdf(self):
        for i, d in enumerate(self.x):
            self.cdf[i] = np.sum(self.pdf[:i])
        return self.cdf
