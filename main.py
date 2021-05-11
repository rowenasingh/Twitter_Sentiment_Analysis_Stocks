import nltk
import numpy
import tweets

if __name__ == '__main__':
    stock = input("Enter the stock you would like to analyze (Stock Symbol e.g. AAPL, TSLA): ")
    days = input("Enter how many days you would like to analyze it for (0-21): ")
    days = int(days)
    if days < 0 or days > 21:
        print("Invalid # of days entered")
        exit(1)
    stock = str(stock)
    tweets = tweets.getTweetsWrapper(stock, days)
    if len(tweets) == 0:
        print("No relevant tweets could be found for this stock. Try again with a higher # of days?")
        exit(1)

    a_file = open("test.txt", "w", encoding="utf-8")
    for tweet in tweets:
        a_file.write(tweet)
        a_file.write("\n")
    ##############################################################
    # Setting up
    # Required modules : TextBlob, pandas_datareader, sklearn
    try:
        import pandas_datareader as web
    except ModuleNotFoundError:
        print("pandas_datareader module not found, pandas_datareader is needed to obtain training data.")
        ans = input("Do you want the script to install it for you (Y/N) (Have to re-run the program after) ?")
        if ans is not None:
            if ans.upper() == 'Y':
                print(ans)
                import subprocess
                import sys

                subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas-datareader"])


            elif ans.upper() == "N":
                print("Terminating program...")
                exit(1)
            else:
                print("Invalid Answer")
                exit(1)

    ##################################################################################

    # Textblob - pretrained sentiment of the tweets

    from textblob import TextBlob
    import pandas as pd
    import re


    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity


    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    dfTB = pd.DataFrame([text for text in tweets], columns=['tweets'])
    dfTB['tweets'] = tweets
    dfTB['Subjectivity'] = dfTB['tweets'].apply(getSubjectivity)
    dfTB['Polarity'] = dfTB['tweets'].apply(getPolarity)


    ###############################################################

    # Trained with custom twitter data found online using sklearn
    # source - https://www.kaggle.com/yash612/stockmarket-sentiment-dataset

    stock_data = pd.read_csv('./dataset/stock_data.csv')
    features = stock_data['Text']
    labels = stock_data['Sentiment']

    # cleaning data
    processed_features = []
    for i in range(0, len(features)):
        processed_feature = re.sub(r'\W', ' ', str(features[i]))
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
        processed_feature = re.sub(r'^b\s+', '', processed_feature)
        processed_feature = processed_feature.lower()
        processed_features.append(processed_feature)

    ########################################################################
    # Imports
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier

    from sklearn.model_selection import train_test_split

    from nltk.corpus import stopwords
    if stopwords: print("Stopwords already up-to-date\n")
    else:
        nltk.download('stopwords')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report, accuracy_score
    import time
    from sklearn.datasets import make_blobs
    # import matplotlib.pyplot as plt
    import datetime as DT

    ########################################################################
    vectorizer = TfidfVectorizer(max_features=10000, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    processed_features = vectorizer.fit_transform(processed_features).toarray()

    X_train, X_test, Y_train, Y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

    # Random Forest model
    start = time.time()
    print("Using TfidfVectorizer and Random Forest model:")
    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train, Y_train)
    predictions = text_classifier.predict(X_test)
    print(classification_report(Y_test, predictions))
    print("Accuracy using Random Forest:", accuracy_score(Y_test, predictions))
    end = time.time()
    print("\nTime Elapsed: %.4f seconds\n" % (end-start))
    ##########################################################################

    # SVM model
    start = time.time()
    print("Using TfidfVectorizer and SVM:")
    text_classifier = SGDClassifier(alpha=1e-5, random_state=0, max_iter=10, tol=None)
    text_classifier.fit(X_train, Y_train)
    predictions = text_classifier.predict(X_test)
    print(classification_report(Y_test, predictions))
    print("Accuracy using SVM:", accuracy_score(Y_test, predictions))
    end = time.time()
    print("\nTime Elapsed: %.4f seconds\n" % (end - start))
    #############################################################################

    # Logistic Regression model
    start = time.time()
    print("Using TfidfVectorizer and Logistic Regression:")
    text_classifier = LogisticRegression(solver='saga', random_state=0)
    text_classifier.fit(X_train, Y_train)
    predictions = text_classifier.predict(X_test)
    print(classification_report(Y_test, predictions))
    print("Accuracy using Logistic Regression:", accuracy_score(Y_test, predictions))
    end = time.time()
    print("\nTime Elapsed: %.4f seconds\n" % (end - start))
    ##########################################################################

    # Logistic Regression comes out most efficient in terms of accuracy so we will use this to predict the tweets sentiment
    # now we will change the tweets (string array to vectors) and predict
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tweetsVector = vectorizer.fit_transform(tweets).toarray()
    X, y = tweetsVector.shape
    x_blob, y_blob = make_blobs(n_samples=len(tweets), centers=2, n_features=y, random_state=0)
    text_classifier = LogisticRegression(solver='saga', random_state=0, max_iter=10000)
    text_classifier.fit(x_blob, y_blob)

    predicted = text_classifier.predict(tweetsVector)
    # change all zeros to -1 because it is binary classification
    predicted = predicted.tolist()
    for i in range(len(predicted)):
        if predicted[i] == 0:
            predicted[i] = -1
    # predicted for scatters
    # dfTB['Polarity'] as Scatters
    # stock price as a line

    polarity = []
    for i in range(len(dfTB)):
        polarity.append(dfTB['Polarity'][i])
    import statistics
    sklearnSentimentAvg = statistics.mean(predicted)
    textBlobSentimentAvg = statistics.mean(polarity)
    print("Trained and predicted polarity average:", sklearnSentimentAvg)
    print("Polarity using TextBlob average:", textBlobSentimentAvg)


    today = DT.date.today()
    date = today - DT.timedelta(days=days)

    stock_prices = web.DataReader(stock, "yahoo", DT.datetime(date.year, date.month, date.day), DT.datetime(today.year, today.month, today.day))


    print("Stock information:\n", stock_prices)
    firstClosePrice = stock_prices.iat[0, stock_prices.columns.get_loc('Adj Close')]
    latestClosePrice = stock_prices.iat[-1, stock_prices.columns.get_loc('Adj Close')]
    print("Adjusted Closing Price in the beginning of analysis:", firstClosePrice)
    print("Adjusted Closing Price in the last day of analysis:", latestClosePrice)
    diff = latestClosePrice - firstClosePrice
    print("Difference in Adjusted Closing Price:", diff)















# This is used to print the training results for empirical analysis, feel free to use or not use it.
# fScores is an array of f1 scores, nameArray is the name of the ML functions (i.e. logistic regression, etc)
# These arrays must be the same length
# def printTrainingResults(fScores, nameArray):
#     for i in range(len(nameArray)):
#         print("Using %s we obtained an F1 score of %s" % (nameArray[i], fScores[i]))


# Given the tweets the machine learning should classify each tweet into a bullish or bearish category.
# Your job here is to pass in an array of percentages (#bullish tweets / (#bullish tweets + #bearish tweets))
# These arrays must be the same length
# def printSentimentAnalysis(sentimentValues, nameArray):
#     for i in range(len(nameArray)):
#         sv = sentimentValues[i]
#         name = nameArray[i]
#         if sv > .75:
#             print("%s predicts a STRONG bullish sentiment w/ value %s\n" % (name, sv))
#         elif sv > .5:
#             print("%s predicts a bullish sentiment w/ value %s\n" % (name, sv))
#         elif sv > .25:
#             print("%s predicts a bearish sentiment w/ value %s\n" % (name, sv))
#         else:
#             print("%s predicts a STRONG bearish sentiment w/ value %s\n" % (name, sv))
