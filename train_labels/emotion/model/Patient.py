import numpy as np


class People:
    # status
    NORMAL = 0
    INFECTION = NORMAL+1
    RECOVER = INFECTION+1
    DIE = RECOVER+1
    # idx
    UID = 0
    DAY = UID+1
    STATUS = DAY+1

    def __init__(self, uid, period):
        """
        tracking the status of a people in the period
        the date and status will both marked.
        all the people will as normal status marked.
        :param period:
        """
        self.uid = ''
        # self.days = []
        self.status = self.pdf = self.cdf = np.zeros(np.shape(period))
        self.uid = uid
        for i, d in enumerate(period):
            # print('i:{} uid: {}, d:{}, status:{}'.format(i, uid, d, self.NORMAL))
            self.status[i] = self.NORMAL
            self.pdf[i] = self.status[i]
            self.cdf[i] = np.sum(self.pdf[:i])
            # self.days.append(d)
        self.status = np.asarray(self.status)
        # self.days = np.asarray(self.days)
        # print(self.status.shape)

    def update(self, idx):
        self.status[idx] += 1
        self.pdf[idx] = self.status[idx]
        # self.label[idx] = label
        # self.cdf[idx] += self.pdf[:idx]
        # print('People.update-> idx:{}, status:{}, pdf:{}, cdf:{}'.format(idx, self.status[idx], self.pdf[idx], self.cdf[idx]))
        # print('People.update-> idx:{}, status:{}, pdf:{}'.format(idx, self.status[idx], self.pdf[idx]))
        return self.status[idx]

    def infection(self, day):
        """
        1. find index of this day
        2. update status
        :return:
        """
        pass

    def recover(self, day):
        """
        1. find index of this day
        2. update status to recover
        :param day:
        :return:
        """
        # idx =
        pass

    def died(self):
        """
        1. find index of this day
        2. update status
        :return:
        """
        pass


class Patient(People):

    def __init__(self, uids, period):
        self.patients = []
        self.uids = uids
        self.periods = period
        for i, uid in enumerate(self.uids):
            p = People(uid, self.periods)
            self.patients.append(p)
        self.patients = np.asarray(self.patients)
        # print('patients: {}'.format(self.patients[0].status[:10,:]))

    def update(self, uid, day):
        """
        1. find index by uid
        2. find index by day
        3. get current status
        4. update new status
        :param uid:
        :param day:
        :return:
        """
        idx = self.uids.where(uid)
        print('idx:{}'.format(idx))
        pass
