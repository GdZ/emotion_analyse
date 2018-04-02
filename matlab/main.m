clc;
% id
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
F_URL           = 15;


tweet = init();

%draw(tweet);
compute(tweet);
labels(tweet);
