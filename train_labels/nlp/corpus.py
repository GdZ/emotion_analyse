from tweet import *


class Corpus():

    def __init__(self, train_file):
        self.train_file = train_file
        self.gold_file = ""
        self.predict_file = ""

        self.train_set = []
        self.emotion = {0: 'sad', 1: 'joy', 2: 'disgust', 3: 'surprise', 4: 'anger', 5: 'fear'}

        self.total_tp = self.total_tn = self.total_fp = self.total_fn = 0
        self.total_precision = self.total_recall = 0
        self.micro_f1 = self.macro_f1 = 0
        self.performance = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], 4: [0, 0, 0], 5: [0, 0, 0]}

    def read_corpus(self):
        # read training data
        fd = open(self.train_file)
        line = fd.readline()
        while (line != ''):
            tweet = Tweet(line.strip('\n'))
            self.train_set.append(tweet)
            line = fd.readline()
        fd.close()
        print("Successfully read corpus!")

    # Logger.i('Successfully read corpus!')

    def getTrainSet(self):
        return self.train_set

    def readGold(self, gold_file):
        self.gold_file = gold_file

        # read gold labels
        file = open(gold_file)
        t = 0
        line = file.readline()
        while (line != ''):
            tweet = self.train_set[t]
            tweet.setGold(line.strip('\n'))
            line = file.readline()
            t += 1

        file.close()
        print("Successfully read gold labels!")

    # Logger.i('Successfully read corpus!')

    def readPrediction(self, predict_file):
        self.predict_file = predict_file

        # read predicted labels
        file = open(predict_file)
        t = 0
        line = file.readline()
        while (line != ''):
            tweet = self.train_set[t]
            tweet.setPredict(line.strip('\n'))
            line = file.readline()
            t += 1
        file.close()
        print("Successfully read predicted labels!")

    # Logger.i('Successfully read predicted labels!')

    def featureExtraction(self):
        for tweet in self.train_set:
            tweet.featureExtraction()

    def evaluation(self):
        # for each emotion, count tp, fp, tn, fn and calculate its precision and recall seperatly
        for e in range(len(self.emotion)):
            tp = tn = fp = fn = 0
            accuracy = precision = recall = f1 = 0
            # count tp, fp, tn, fn for e
            for i in range(len(self.train_set)):
                tweet = self.train_set[i]
                gold = tweet.getGold()
                predict = tweet.getPredict()
                if gold == predict and gold == e:
                    tp += 1
                elif predict == e and gold != e:
                    fp += 1
                elif gold == e and predict != e:
                    fn += 1
                else:
                    tn += 1
            # print tp, fp, tn, fn
            # x = raw_input()

            if (tp + fp) != 0:
                precision = 1.0 * tp / (tp + fp) * 100
            if (tp + fn) != 0:
                recall = 1.0 * tp / (tp + fn) * 100
            if (precision + recall) != 0:
                f1 = 2.0 * precision * recall / (precision + recall)
            # saving performance for each emotion
            self.performance[e] = [precision, recall, f1]

            # adding up together for further calculation
            self.total_tp += tp
            self.total_tn += tn
            self.total_fp += fp
            self.total_fn += fn
            self.total_precision += precision
            self.total_recall += recall
        # print

        # micro_f1 calculation
        if (self.total_tp + self.total_fp) != 0:
            micro_precison = 1.0 * self.total_tp / (self.total_tp + self.total_fp) * 100
        if (self.total_tp + self.total_fn) != 0:
            micro_recall = 1.0 * self.total_tp / (self.total_tp + self.total_fn) * 100
        if (micro_precison + micro_recall != 0):
            self.micro_f1 = 2.0 * micro_precison * micro_recall / (micro_precison + micro_recall)

        # macro_f1 calculation
        macro_precison = self.total_precision / len(self.emotion)
        macro_recall = self.total_recall / len(self.emotion)
        if (macro_precison + macro_recall) != 0:
            self.macro_f1 = 2.0 * macro_precison * macro_recall / (macro_precison + macro_recall)

    def print_result(self):
        for e in range(len(self.emotion)):
            print self.emotion[e], '	Precision:', self.performance[e][0], '	Recall:', self.performance[e][
                1], '	F1:', self.performance[e][2]
        print "Micro F1 Score:", self.micro_f1
        print "Macro F1 Score:", self.macro_f1
