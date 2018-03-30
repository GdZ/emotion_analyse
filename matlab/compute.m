function compute(tweet)
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
    id          = tweet{1, F_ID};
    uid         = tweet{1, F_USER_ID};
    type        = tweet{1, F_TYPE};
    sentiment   = tweet{1, F_SENTIMENTLABEL};
    labels      = tweet{1, F_LABELSVALUE};
    uname       = tweet{1, F_USERNAMETWEET};
    text        = tweet{1, F_TEXT};
    reply       = tweet{1, F_IS_REPLY};
    retweet     = tweet{1, F_IS_REPLY};
    nr_favor    = tweet{1, F_NR_FAVOR};
    nr_reply    = tweet{1, F_NR_REPLY};
    nr_retweet  = tweet{1, F_NR_RETWEET};
    date        = tweet{1, F_DATE};
    time        = tweet{1, F_TIME};
    url         = tweet{1, F_URL};

    % --------------------------------------------------------------------------
    % info in a day
    % date
    % count_user        : count of user
    % count_tweet       : count of tweet
    % count_favor       : count of favor
    % count_retweet     : count of retweet
    % count_repy        : count of repy
    days = [];
    % count
    count_user = [];
    count_tweet = [];
    count_favor = [];
    count_retweet = [];
    count_reply = [];
    % delta of number
    delta_user = [];
    delta_tweet = [];
    delta_favor = [];
    delta_retweet = [];
    delta_reply = [];
    % sum of count
    sum_user = [];
    sum_tweet = [];
    sum_favor = [];
    sum_retweet = [];
    sum_reply = [];
    % average label
    label_user = [];
    label_tweet = [];

    % --------------------------------------------------------------------------
    % compute
    % date_ds - date
    % date_idx - index of a day in array date
    % date_pos - index of a day in array date_ds
    [date_ds, date_idx, date_pos] = unique(date);
    [uid_ds, uid_idx, uid_pos] = unique(uid);

    % --------------------------------------------------------------------------
    BREIT = length(date_ds);

    % --------------------------------------------------------------------------
    % caculate the number
    for i = 1:length(date_ds)
        days = [ days date_ds(i,:) ];
        index = find(date_ds(i) == date);

        % count the number of tweet each day
        count_tweet(i,:) = length(index);
        sum_tweet(i,:) = sum(count_tweet);

        % count number of user each day
        count_user(i,:) = length(unique(uid(index',:)));
        sum_user(i,:) = sum(count_user);

        % count number of favor, retweet, reply each day
        count_favor(i,:) = sum(nr_favor(index',:));
        count_retweet(i,:) = sum(nr_retweet(index',:));
        count_reply(i,:) = sum(nr_retweet(index',:));
        sum_favor(i,:) = sum(count_favor);
        sum_retweet(i,:) = sum(count_retweet);
        sum_reply(i,:) = sum(count_reply);

        if 1 == i
            delta_tweet(i,:) = 0;
            delta_user(i,:) = 0;
            delta_favor(i,:) = 0;
            delta_retweet(i,:) = 0;
            delta_reply(i,:) = 0;
        else
            delta_tweet(i,:) = count_tweet(i,:) - count_tweet((i-1),:);
            delta_tweet(i,:) = count_tweet(i,1) - count_tweet(i-1,1);
            delta_user(i,:) = count_user(i,:) - count_user(i-1,:);
            delta_favor(i,:) = count_favor(i,:) - count_favor(i-1,:);
            delta_retweet(i,:) = count_retweet(i,:) - count_retweet(i-1,:);
            delta_reply(i,:) = count_reply(i,:) - count_reply(i-1,:);
        end
        % label
        label_tweet(i,:) = sum(labels(index',:))/length(index);
    end

    % --------------------------------------------------------------------------
    for i = 1:length(uid_ds)
        index = find(uid_ds(i) == uid);
        label_user(i,:) = sum(labels(index',:))/length(index);
    end


    % --------------------------------------------------------------------------
    % draw picture
    if DEBUG
        fig1 = figure(); hold on; grid on;
        % days - number of tweet
        plot(days(1,1:BREIT), count_tweet(1:BREIT,1), 'o-', 'color', 'r');
        plot(days(1,1:BREIT), count_user(1:BREIT,1), 'o-', 'color', 'b');
        plot(days(1,1:BREIT), count_favor(1:BREIT,1), '.-', 'color', 'r');
        plot(days(1,1:BREIT), count_retweet(1:BREIT,1), 'o-', 'color', 'b');
        plot(days(1,1:BREIT), count_reply(1:BREIT,1), 'x-', 'color', 'y');
    end

    if DEBUG
        % days - number of user
        fig2 = figure(); hold on; grid on;
        plot(days(1,1:BREIT), delta_tweet(1:BREIT,1), 'o-', 'color', 'r');
        plot(days(1,1:BREIT), delta_user(1:BREIT,1), 'o-', 'color', 'b');
        plot(days(1,1:BREIT), delta_favor(1:BREIT,1), '.-', 'color', 'r');
        plot(days(1,1:BREIT), delta_retweet(1:BREIT,1), 'o-', 'color', 'b');
        plot(days(1,1:BREIT), delta_reply(1:BREIT,1), 'x-', 'color', 'y');
    end

    if DEBUG
        % days - number of favor, retweet, reply
        fig3 = figure(); hold on; grid on;
        plot(days(1,1:BREIT), sum_tweet(1:BREIT,1), 'o-', 'color', 'r');
        plot(days(1,1:BREIT), sum_user(1:BREIT,1), 'o-', 'color', 'b');
        plot(days(1,1:BREIT), sum_favor(1:BREIT,1), '.-', 'color', 'r');
        plot(days(1,1:BREIT), sum_retweet(1:BREIT,1), 'o-', 'color', 'b');
        plot(days(1,1:BREIT), sum_reply(1:BREIT,1), 'x-', 'color', 'y');

    end

    % --------------------------------------------------------------------------
    BREIT = length(date_ds);

    % --------------------------------------------------------------------------
    % draw picture
    % days - label of tweet
    if DEBUG
        fig4 = figure(); hold on; grid on;
        plot(days, label_tweet, 'x-', 'color', 'r');
    end

    % days - label of user
    if DEBUG
        fig5 = figure(); hold on; grid on;
        plot(1:length(uid_ds), label_user, '.', 'color', 'r');
    end

end
