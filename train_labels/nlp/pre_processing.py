#!/usr/bin/python
# -*- coding: UTF-8 -*-
import re, string

def define_regex():
	#the regular expression of emoji
	emoticons_str = r"""
		(?:
			[:=;] # Eyes
			[oO\-]? # Nose (optional)
			[D\)\]\(\]/\\OpP] # Mouth
		)"""

	#the regular expression of HTMLtags, @personName, URLs, numbers and 'NEWLINE' which need to be deleted
	regex_substr = [
		r'<[^>]+>', # HTML tags
		r'(?:@[\w_]+)', # @personName
		r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
		r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
		r'NEWLINE' #the special word 'NEWLINE'
	]

	#the regular expression of words and anything else which need to be left
	regex_str = [
			emoticons_str,
			r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
			r'(?:[\w_]+)', # other words
			r'(?:\S)' # anything else
	]

	__tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
	__del_re = re.compile(r'('+'|'.join(regex_substr)+')', re.VERBOSE | re.IGNORECASE)
	__hash_re = re.compile(r'(?:\#+)([\w_]+[\w\'_\-]*[\w_]+)') #Hashtags
	__punc_re = re.compile(r'[%s]' % re.escape(string.punctuation))
	__puncn_re = re.compile(r'[£¡£¿?££¡ç£¥^&*\'\"£¨£©£ª£«£¬£­£¯£º£»£¼£½£¾@£Û£Ý\¡¢£ß£û£ý|¡«~`¡¶¡·¡¸¡¹¡º¡»¡¾¡¿¡²¡³¡¼¡½¨C¡ª¡®¡¯¡°¡±¡­¡£.]', re.VERBOSE | re.IGNORECASE)
	__emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
	return __tokens_re , __del_re , __hash_re , __punc_re , __puncn_re , __emoticon_re

def tokenize(tweet): # delete sth. not to be needed & find sth. to be needed
	__tokens_re, __del_re, __hash_re, __punc_re, __puncn_re,__emoticon_re =  define_regex()
	tweet = __del_re.sub(r'', tweet) #delete HTMLtags, @personName, URLs, numbers and 'NEWLINE'
	tweet = __hash_re.sub(r'\1', tweet) #delete hashtag but leaving the word after hashtag
	tweet = __puncn_re.sub(r'', tweet) #delete chinese punctuation
	tweet = __punc_re.sub(r'', tweet) #delete english punctuation
	return __tokens_re.findall(tweet) , __emoticon_re
        
def preprocess(tweet, lowercase=False): # tokenize
	tokens, __emoticon_re = tokenize(tweet)
	
	#transform capital into lowercase except emoji tokens
	if lowercase: 
		tokens = [token if __emoticon_re.search(token) else token.lower() for token in tokens]
	return tokens
			