import numpy

#Multi-Class Perceptron
emotion = {'sad':0, 'joy':1, 'disgust':2 , 'surprise':3, 'anger':4, 'fear':5}

def matrix_dot(features, weight): #vector: list; weight: dict
	y = [0 for i in range(6)]
	for e in range(len(emotion)):
		for f in features:
			if weight[e]==None:
				weight[e] = {f:0}
			if not weight[e].has_key(f):
				weight[e][f] = 0
			w = weight[e].get(f)
			y[e] += w
	return y

def train(iteration, train_set):
	weight = dict.fromkeys(range(len(emotion)))
	#print weight

	for i in range(iteration):
		#print('Iteration:', i)
		for tweet in train_set:
			features = tweet.getFeatures()
			gold = tweet.getGold()
			y = matrix_dot(features, weight)
			
			max = numpy.max(y)
			for e in range(0,len(emotion)):
				if y[e] == max:
					predict = e

			#print predict, gold
			if predict != gold:
				for f in features:
					weight[gold][f] = weight[gold][f] + 1
					weight[predict][f] = weight[predict][f] - 1
			#print gold, weight[gold]
			#print predict, weight[predict]
			#raw_input()

	#output weight to file
	file = open('weight.txt','w')
	for e in range(len(emotion)):
		file.write(str(e)+'\n')
		file.write(str(weight[e])+'\n\n\n\n\n')
	file.close()

	print('Multi-calss perceptron successfully trained!')
	return weight

def test(test_set, weight):
	for tweet in test_set:
		features = tweet.getFeatures()
		gold = tweet.getGold()
		y = matrix_dot(features, weight)
			
		max = numpy.max(y)
		for e in range(0,len(emotion)):
			if y[e] == max:
				predict = e
		tweet.setPredict(predict)
	print('Successfully tested!')

