function draw(tweet)

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
	F_URL           = 15;


	% draw some figure
 	% plot(tweet{1, F_ID}, tweet{1, F_LABELSVALUE}, '.');
 	% number of reply
    plot(tweet{1, F_USER_ID}, tweet{1, F_NR_REPLY}, 'x');
    grid on;
    % number of retweet
    hold on;
    plot(tweet{1, F_USER_ID}, tweet{1, F_NR_RETWEET}, '.');
    % number of favor
    plot(tweet{1, F_USER_ID}, tweet{1, F_NR_FAVOR}, 'o');
    % number of user by date
    plot(tweet{1, F_DATE}, tweet{1, F_USER_ID}, '-o-');

    % -----------------------------
    % label and F_ID
    plot(tweet{1, F_ID}, tweet{1, F_LABELSVALUE}, '*');
    % label and F_USER_ID
    plot(tweet{1, F_USER_ID}, tweet{1, F_LABELSVALUE}, '*');

end
