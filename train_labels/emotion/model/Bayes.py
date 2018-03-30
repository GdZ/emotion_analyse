# coding: utf-8


def split_text_by_label(vector_text, label_list):
    split_text = {}
    for i in range(len(label_list)):
        if len(vector_text[i]):
            current_label = label_list[i]
            if current_label not in split_text:
                split_text[current_label] = []
                split_text[current_label].append(vector_text[i])
            else:
                split_text[current_label].append(vector_text[i])
        else:
            pass
    return split_text


# calculate probability of a label/tested
def prior_probability(label_list):
    total_count = len(label_list)
    prior_prob = {}
    labels = ['love', 'disgust', 'trust', 'happy', 'fear', 'sad', 'anger', 'surprise']
    # for label in labels:
    #     count = 0
    #     for current_label in label_list:
    #         if label == current_label:
    #             count += 1
    #     prior_prob[label] = count / total_count
    #
    prior_prob = {'love': 0.1, 'disgust': 0.1, 'trust': 0.1, 'happy': 0.3, 'fear': 0.1, 'sad': 0.1, 'anger': 0.1,
                  'surprise': 0.1}
    return labels, prior_prob


def calculate_prob(split_text, prior_prob):
    # format of token_given_label
    # key:labels
    # value:token count given label
    # usages: given a certain label, calculate how many times one feature(token) occurs
    token_given_label = {}
    # format of token_count
    token_count = {}
    # format of total_token
    # key:feature
    # value:total_tokens
    # usage: how many tokens in the corpus
    total_token = {}
    for (label, texts) in split_text.items():
        token_given_label[label] = {}
        token_count[label] = {}
        for text in texts:
            text_split = text.split()
            for token in text_split:
                if token not in total_token:
                    total_token[token] = 1
                else:
                    pass
                if token not in token_count[label]:
                    token_count[label][token] = 1
                else:
                    pass
                # add token to prob_t_g_l
                if token not in token_given_label[label]:
                    token_given_label[label][token] = 1
                else:
                    token_given_label[label][token] += 1
        # print(prob_token_given_label[label])

    feature_label = {}
    for (label, tokens) in token_given_label.items():
        feature_label[label] = {}
        for (token, count) in total_token.items():
            if token in tokens:
                feature_label[label][token] = (tokens[token] + 1) / (len(token_count[label]) + len(total_token)) \
                                              * prior_prob[label]
            else:
                feature_label[label][token] = 1/(len(token_count[label]) + len(total_token)) \
                                              * prior_prob[label]

    return feature_label


def naive_bayes(file_train_text, file_train_label, file_test):
    # read file to train and test list
    train_text = []
    train_label = []
    for (line, label) in zip(file_train_text, file_train_label):
        if not line.split():
            pass
        else:
            train_label.append(label.strip())
            train_text.append(line.strip())
    test_corpus = []
    result_label = []
    for line in file_test:
        test_corpus.append(line.strip())
    # Split
    text = split_text_by_label(train_text, train_label)
    # get prior probability
    labels, prior_prob = prior_probability(train_label)

    # use feature_label to generate predicted probability
    feature_label = calculate_prob(text, prior_prob)
    for line in test_corpus:
        line_split = line.split(" ")
        prob = 0
        label = ""
        for (current_label, tokens) in feature_label.items():
            prob_current = 1
            for token_line in line_split:
                if token_line not in tokens:
                    pass
                else:
                    prob_current = tokens[token_line] * prob_current
            if prob_current > prob:
                prob = prob_current
                label = current_label
        result_label.append(label)

    # output
    count = 0
    for i in range(len(result_label)):
        if result_label[i] == train_label[i]:
            count += 1
    acc = len(result_label)
    return acc