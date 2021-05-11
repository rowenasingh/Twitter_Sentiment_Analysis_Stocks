import os
import tweepy as tw
import datetime as DT
import re
# import demoji
# import pandas as pd
consumer_key = "LnlfB6qRrdmDS6LfgRFwHlLsF" # NOTE THESE KEYS WILL BE DISABLED AFTER 5/31, PLEASE APPLY FOR TWITTER API DEV ACCESS INDEPENDENTLY
consumer_secret = "RPEarNKAfLWzejlC6bg8NNvBsJAbQgBkJ5zE7eVWuCAYAdSgrb"
access_token = "3429998675-82MVvv0xtMqAcdoIuTDqTDvCJptSNigUBfSgD4T"
access_token_secret = "VV4eOIno767v8zColXNUySs9cPVXnLp7U8HgkDAs0axyG"
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)



def getTweets(ticker, date):
    search_words = "$" + ticker + " AND ((I think) OR (bullish) OR (bearish) OR (bought) OR (sold) OR (buy) OR (sell) OR " \
                            "(dip) OR (gains) OR (sad) OR (happy) OR (feel) OR (makes me) OR (I'm) OR (I am))"
    # "AND ((I think) OR (bullish) OR (bearish) OR (call) OR (put) OR (buy) OR (sell) OR (dip) OR (gains) OR (sad) OR (happy) OR (feel) OR (makes me) OR (I'm) OR (I am))"
    search_words += " -filter:retweets" #ignore retweets to avoid skewed data
    tweets = tw.Cursor(api.search,
                       q=search_words,
                       lang="en",
                       tweet_mode='extended',
                       since=date).items()
    return tweets
# AAPL AND ((I think) OR (bullish) OR (bearish) OR (call) OR (put) OR (buy) OR (sell) OR (dip) OR (gains) OR (sad) OR (happy) OR (feel) OR (makes me) OR (I'm) OR (I am))

# Call with a ticker and the # of days to analyze tweet data, numdays between 0 and 21
# Returns an array containing the tweets, if there is no tweets it is an empty array
def getTweetsWrapper(ticker, numDays):
    today = DT.date.today()
    date = today - DT.timedelta(days=numDays)
    return parseTweets(getTweets(ticker, date))


def parseTweets(tweets):
    newTweets = []
    for tweet in tweets:
        betterTweet = removeGarbage(tweet.full_text)
        if betterTweet is not None and len(betterTweet) is not 0:
            newTweets.append(betterTweet)
    return spamProtection(newTweets)
# Press the green button in the gutter to run the script.


def removeGarbage(tweet): # Removes links and non alphanumeric characters
    # We want to remove stuff the ML cant process, so @'s, links, maybe emojis?
    tweet = str(tweet)
    tweet = re.sub(r'http\S+', '', tweet) # Pretty easy to remove URLS, just an import
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)

    tweet = tweet.replace("\n", " ")
    tweet = tweet.replace("\t", " ")
    tweet = tweet.replace("\r", " ")


    tweet = deEmojify(tweet)
    tweet = ' '.join(s for s in tweet.split(" ") if not shouldRemove(s))
    # tweet = ' '.join(s for s in tweet.split() if not any(not c.isalnum() for c in s))
    # tweet = ' '.join(s for s in tweet.split() if not any(c == '@' for c in s))
    tweet = re.sub(' +', ' ', tweet)
    return tweet.strip()


def shouldRemove(word):
    string_check= re.compile('[@_!#$%^&*()<>?/\|}{~:]')
    if string_check.search(word) is None:
        return False
    else:
        return True

# def deEmojify(inputString):
#     return inputString.encode('ascii', 'ignore').decode('ascii')

def deEmojify(text):
    return text.encode('ascii', 'ignore').decode('ascii')
    # demoji.replace_with_desc(string=text, repl="")
    # emoji_pattern = re.compile("["
    #                            u"\U0001F600-\U0001F64F"  # emoticons
    #                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    #                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
    #                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    #                            u"\U00002702-\U000027B0"
    #                            u"\U000024C2-\U0001F251"
    #                            "]+", flags=re.UNICODE)
    # return emoji_pattern.sub(r'', text)
    # regrex_pattern = re.compile(pattern = "["
    #     u"\U0001F600-\U0001F64F"  # emoticons
    #     u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    #     u"\U0001F680-\U0001F6FF"  # transport & map symbols
    #     u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    #                        "]+", flags = re.UNICODE)
    # return regrex_pattern.sub(r'',text)


# Removes repeat tweets, i.e. prevents spam. Only one spam will be counted.
def spamProtection(tweets):
    newTweets = []
    for tweet in tweets:
        if tweet not in newTweets:
            newTweets.append(tweet)
        # else:
        #     print("Removed a tweet!", tweet)
    return newTweets

#
# if __name__ == '__main__':
#     ticker = "AAPL"
#     tweets = getTweetsWrapper(ticker, 0)
#     print(tweets)
#     print(len(tweets))
#     #parsedTweets = parseTweets(tweets)
#     #print(parsedTweets)
#     #print(len(parsedTweets))
#     # for twit in twits:
#     #     print(twit)
#         # body = twit.body
#         # print(body)
#     # print(test)
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/


# getTrainingData