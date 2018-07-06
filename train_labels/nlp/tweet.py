from pre_processing import *

emotion = {'sad':0, 'joy':1, 'disgust':2 , 'surprise':3, 'anger':4, 'fear':5}

class Tweet():
	def __init__(self, line):
		predict_e, tweet = line.split('	')
		self.tweet = tweet
		self.predict = emotion[predict_e]
		self.gold = -1
		self.tokens = preprocess(self.tweet)
		self.features = []
	
	
	def featureExtraction(self):
		words = []
		bigrams = []
		
		pre = 'SOS'
		for t in self.tokens :
			words.append('word=' + t)
			bigrams.append('bigram='+ t + '/' + pre)
			pre = t
		
		self.features = words + bigrams
		return self.features
	
	def getTweet(self):
		return self.tweet

	def setPredict(self, predict):
		self.predict = predict

	def getPredict(self):
		return self.predict
	
	def setGold (self, gold_e):
		self.gold = emotion[gold_e]
		
	def getGold(self):
		return self.gold
	
	def getFeatures(self):
		return self.features