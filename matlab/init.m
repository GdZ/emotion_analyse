function [tweet] = init()

    clear;
    clc;
    tweet = {};

    CSV_FILE = 'data/marked.csv';

    ROW_START = 2;
    ROW_END = 9552;

	% idx
	F_ID            = 1;
	F_USER_ID       = 2;
	F_TYPE          = 3;
	F_SENTIMENTLABEL = 4;
	F_LABELSVALUE   = 5;
	F_USERNAMETWEET = 6;
	F_TEXT          = 7;
    F_IS_REPLY      = 8;
    F_IS_RETWEET    = 9;
    F_NR_FAVOR 		= 10;
    F_NR_REPLY 		= 11;
    F_NR_RETWEET 	= 12;
	F_DATE    		= 13;
	F_TIME    		= 14;
    F_HAS_MEDIA     = 15;
    F_MEDIAS0       = 16;
	F_URL           = 17;

	% import data to orignal_data
	orignal_data = importfile(CSV_FILE, ROW_START, ROW_END);

	% store to variables
	id          = table2array(orignal_data(:, F_ID));
	user_id     = table2array(orignal_data(:, F_USER_ID));
	type        = table2array(orignal_data(:, F_TYPE));
	sentiment_label = table2array(orignal_data(:, F_SENTIMENTLABEL));
	labels_value = table2array(orignal_data(:, F_LABELSVALUE));
	user_name_tweet = table2array(orignal_data(:, F_USERNAMETWEET));
	text        = table2array(orignal_data(:, F_TEXT));
	is_repy 	= table2array(orignal_data(:, F_IS_REPLY));
	is_retweet 	= table2array(orignal_data(:, F_IS_RETWEET));
	nr_favor 	= table2array(orignal_data(:, F_NR_FAVOR));
	nr_reply 	= table2array(orignal_data(:, F_NR_REPLY));
	nr_retweet  = table2array(orignal_data(:, F_NR_RETWEET));
	date  		= table2array(orignal_data(:, F_DATE));
	time  		= table2array(orignal_data(:, F_TIME));
    has_media   = table2array(orignal_data(:, F_HAS_MEDIA));
    meidas_0    = table2array(orignal_data(:, F_MEDIAS0));
	url         = table2array(orignal_data(:, F_URL));

	tweet = {id, user_id, type, sentiment_label, labels_value, user_name_tweet, text, is_repy, is_retweet, nr_favor, nr_reply, nr_retweet, date, time, url};


end
