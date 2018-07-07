import numpy as np


class Label:
    emotion = {'-5':0, '-4':1, '-3':2, '-2':3, '-1':4, '0':5, '1':6, '2':7, '3':8, '4':9, '5':10}
    keys = []
    values = []

    def __init__(self, labels):
        self.keys = np.asarray(labels)
        self.values = np.zeros(np.shape(self.keys))
        # print('keys: {}\nvalues: {}'.format(self.keys.shape, self.values.shape))
        for i, k in enumerate(self.get_keys()):
            # print('i:{}, k:{}'.format(i, str(k[0])))
            self.values[i] = self.emotion[str(k[0])]

    def get_emotions(self):
        return self.emotion

    def get_keys(self):
        return self.keys

    def set_keys(self, keys):
        self.keys = keys

    def get_values(self):
        return self.values

    def set_values(self, values):
        self.values = values
