function labels(tweet)

    % --------------------------------------------------------------------------
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
    F_NR_FAVOR      = 10;
    F_NR_REPLY      = 11;
    F_NR_RETWEET    = 12;
    F_DATE          = 13;
    F_TIME          = 14;
    F_URL           = 15;

    % --------------------------------------------------------------------------
    DEBUG = 1;

    % --------------------------------------------------------------------------
    % prepare
    % id          = tweet{1, F_ID};
    % uid         = tweet{1, F_USER_ID};
    % type        = tweet{1, F_TYPE};
    % sentiment   = tweet{1, F_SENTIMENTLABEL};
    % labels      = tweet{1, F_LABELSVALUE};
    % uname       = tweet{1, F_USERNAMETWEET};
    % text        = tweet{1, F_TEXT};
    % reply       = tweet{1, F_IS_REPLY};
    % retweet     = tweet{1, F_IS_REPLY};
    % nr_favor    = tweet{1, F_NR_FAVOR};
    % nr_reply    = tweet{1, F_NR_REPLY};
    % nr_retweet  = tweet{1, F_NR_RETWEET};
    % date        = tweet{1, F_DATE};
    % time        = tweet{1, F_TIME};
    % url         = tweet{1, F_URL};
    emotion_label      = tweet{1, F_LABELSVALUE};

	idx = 1:9551;

	[mu, sigma] = normfit(emotion_label)
	[count, center] = hist(emotion_label) % draw a histogram

	bar(center, count, 'FaceColor', 'r', 'EdgeColor', 'w');
	box off
	xlim([mu-3*sigma, mu+3*sigma])
	a2 = axes;
	ezplot(@(center)normpdf(center, mu, sigma), [mu-3*sigma, mu+3*sigma])
	set(a2, 'box', 'off', 'yaxislocation', 'right', 'color', 'none')
	title '频数直方图与正态分布密度函数（拟合）'

	grid on;

end
