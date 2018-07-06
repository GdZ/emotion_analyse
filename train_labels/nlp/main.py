from corpus import *

import perceptron

if __name__ == '__main__':
    # Read corpus
    # corp_path = raw_input("Please input the path of the corpus:\n")
    # train_file = raw_input("Please input the filename of the training data:\n")
    # gold_file = raw_input("Please input the filename of the gold label:\n")
    train_file = "trial.csv"
    corpus = Corpus(train_file)
    corpus.read_corpus()

    # Read gold
    gold_file = "trial.labels"
    corpus.readGold(gold_file)

    # feature extraction
    corpus.featureExtraction()

    # Baseline training part.
    iteration = 1
    weight = perceptron.train(iteration, corpus.getTrainSet())

    perceptron.test(corpus.getTrainSet(), weight)

    # Evaluation
    corpus.evaluation()
    corpus.print_result()
