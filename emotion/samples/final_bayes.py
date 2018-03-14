#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
# toy data
tweet_1 = {'emotion': 'anger', 'content': 'mahak MarioDP Su tonto hass papa en la esquina happy josuconnors Te echar de menos Vamos por favor   friends fun intoyou arianagrande songü https://t.co/0fNaE6I672', }
tweet_2 = {'emotion': 'happy', 'content': 'mahak  mahak IamWBCS / O to all the fans cant say it enough WildBoyCrazy cool swag + A}:R', }
tweet_3 = {'emotion': 'happy', 'content': 'mahak KatherineMacKenzie Grad rehearsal tonight. Excitement is in the air. Cant wait to see everyone all done up tomorrow grad besafe -1;-1 n/a', }
tweet_4 = {'emotion': 'happy', 'content': 'lisarowland My body is tired but my head is goin and goin I have no energy at all drained ü https://t.co/L936hIGJqy -1;-1 n/a', }
tweet_5 = {'emotion': 'love', 'content': 'EclipseThe beautiful Mount Assiniboine in British Columbia | Photo by Callum Snape Dream Love Hope Health Peace https://t.co/aOqqlSDwGP-1;-1 n/a http://pbs.twimg.com/media/CW79ChcUoAAGz8G.jpg', }
tweet_6 = {'emotion': 'happy', 'content': 'RoxanaJones About Happiness...NEWLINEvia @StevenAitchison https://t.co/O9aJQpGm4G -1;-1 n/a http://pbs.twimg.com/media/CiSitZLUkAAmR5D.jpg' ,}
tweet_7 = {'emotion': 'sad', 'content': 'mahak Samantharuth prabhu ness fellingto cry unhappiness missingsome1ゐ what can I do -1;-1 n/a ', }
tweet_8 = {'emotion': 'love', 'content': 'LuzMaldonado Rivas memories heart heart family photo picture gift always forever kids mum dad parents siblingsü https://t.co/ECS5vRE2Fe-1;-1 n/a http://pbs.twimg.com/media/CjvigDWWkAAdQJo.jpg', }
tweet_9 = {'emotion': 'happy', 'content': 'MaevaJillson @UnitedAgents more than 6 000 views on my video of SOLs Letter... OMG :D HappyNEWLINEhttps://t.co/cHggx1VxQG -1;-1 n/a', }
tweet_10 = {'emotion': 'sad', 'content': 'Ponyalways Pony swag weeaboo furry emo depressed anime filthyfrank h3h3 zootopia judyhops nickwilde lol wow ü https://t.co/D1MhOaNM86 -1;-1 n/a http://pbs.twimg.com/media/ChtYWzcXIAEnVAG.jpg', }
training_tweets = [tweet_1, tweet_2, tweet_3, tweet_4, tweet_5, tweet_6, tweet_7, tweet_8, tweet_9, tweet_10]

# emotions
emotions = {'happy', 'sad', 'anger', 'love', 'trust', 'fear', 'disgust' , 'surprise'}
features = {}

# initialize emotion_list
emotion_list = {}
for e in emotions:
    emotion_value = {'tokens_count': 0, 'emotions_count': 0, 'emotions_prior_prob': 0}
    emotion_list[e] = emotion_value

features_total = 0
feature_prior_prob = 0.0
emotion_feature_list = []
emotion_feature_count = 0
emotion_feature_prob = 0.0
class_prob = 0
test_word_list  = []

# Emotion features for training and testing
emotion_feature = {}
test_emotion_features = {}
classification = 0.0
# emotion prior
# determine the frequency of each emotion


def emotions_count(training_tweets):
    for tweet in training_tweets:
        tweet_content = tweet['content']
        tweet_emotion = tweet['emotion']
        emotion_list[tweet_emotion]['emotions_count'] += 1
        emotion_list[tweet_emotion]['tokens_count'] += len(tweet_content.split())


emotions_count(training_tweets)


# get list of features and prior features probabilities from the word_list
def word_list():
    with open('word_list.txt', 'r', encoding='utf8') as f:
        for word in f:
            word.lower().replace("\n", "")
            if word in features:
                features[word]['count'] += 1
            else:
                features[word] = {}
                features[word]['count'] = 1
                features[word]['prior_prob'] = 0


word_list()


# get the feature probabilities based on their association with the emotions


def emotions_and_features_counts():
    """
    test
    :return:
    """
    tweets = training_tweets

    for tweet in tweets:
        tweet_tokens = tweet['content']
        tweet_emotion = tweet['emotion']

        tokens = tweet_tokens.split()

        for word in tokens:
            if word in emotion_feature:
                if tweet_emotion in emotion_feature[word]:
                    emotion_feature[word][tweet_emotion]['count'] +=1
                    emotion_feature[word]['total'] += 1
                else:
                    emotion_feature[word].update({tweet_emotion : {'count': 1,'class_prob': 0}})
                    emotion_feature[word]['total'] += 1

            else:
                features[word] = {}

                emotion_feature[word] = {tweet_emotion : {'count': 1,'class_prob': 0}, 'total': 1}
    return emotion_feature


def test_tweets():
    global emotions
    prob_all  = []
    for tweet in test_tweet:
        tokens = tweet['content'].split(' ')
        prob = {}
        for emotion in emotions:
            total_prob = 0
            for token in tokens:
                if token not in emotion_features_smoothing:
                    prob_token = 1.0 / (emotion_list[emotion]['tokens_count'] + len(emotion_features_smoothing))
                elif emotion not in emotion_features_smoothing[token]:
                    prob_token = 1.0 / (emotion_list[emotion]['tokens_count'] + len(emotion_features_smoothing))
                else:
                    prob_token = emotion_features_smoothing[token][emotion]['class_prob']

                total_prob += math.log(prob_token)
            prob[emotion] = total_prob
        print(prob)
        prob_all.append(prob)
    return prob_all


def emotions_and_features_probabilities(emotion_features, smoothing=False):
    for word in  emotion_features:
        for emotion in emotion_features[word]:
            if emotion != 'total':
                if smoothing == True:
                    emotion_features[word][emotion]['class_prob'] = (emotion_features[word][emotion]['count'] +1) * 1.0 /  \
                                                                (emotion_list[emotion]['tokens_count'] + len(emotion_features))
                else:
                    emotion_features[word][emotion]['class_prob'] = (emotion_features[word][emotion]['count']) * 1.0/emotion_list[emotion]['tokens_count']

    return emotion_features
            # tot_class_count += emotion_feature[word][emotion]['count']
            

emotion_features = emotions_and_features_counts()
emotion_features_nonsmoothing = emotions_and_features_probabilities(emotion_features)
emotion_features_smoothing = emotions_and_features_probabilities(emotion_features, True)

test_tweet = [tweet_2]

print(emotion_list)
print(emotion_features_nonsmoothing)
print(emotion_features_smoothing)

prob_all = test_tweets()


#
#
#
#
# def NBCResult(emotion):
#     max(classification)
#     return NBCResult
# print(NBCResult)
#
#
#
#

