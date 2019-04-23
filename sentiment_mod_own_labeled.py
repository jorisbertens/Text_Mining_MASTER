import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk import word_tokenize
import pickle
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

##### This is all part of the combination of 7 classifier to identify the sentiment of a tweet

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        
documents_f = open("pickled_algorithms/own_labeled/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open("pickled_algorithms/own_labeled/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
#featuresets_f = open("pickled_algorithms/featuresets.pickle","rb")
#featuresets = pickle.load(featuresets_f)
#featuresets_f.close()

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

classifier_f = open("pickled_algorithms/own_labeled/originalnaivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

MNB_f = open("pickled_algorithms/own_labeled/MNB.pickle", "rb")
MNB_classifier = pickle.load(MNB_f)
MNB_f.close()

BernoulliNB_f = open("pickled_algorithms/own_labeled/BernoulliNB.pickle", "rb")
BernoulliNB_classifier = pickle.load(BernoulliNB_f)
BernoulliNB_f.close()

LinearSVC_f = open("pickled_algorithms/own_labeled/LinearSVC.pickle", "rb")
LinearSVC_classifier = pickle.load(LinearSVC_f)
LinearSVC_f.close()

NuSVC_f = open("pickled_algorithms/own_labeled/NuSVC.pickle", "rb")
NuSVC_classifier = pickle.load(NuSVC_f)
NuSVC_f.close()

SGDClassifier_f = open("pickled_algorithms/own_labeled/SGDClassifier.pickle", "rb")
SGDClassifier_classifier = pickle.load(SGDClassifier_f)
SGDClassifier_f.close()

LogisticRegression_f = open("pickled_algorithms/own_labeled/LogisticRegression.pickle", "rb")
LogisticRegression_classifier = pickle.load(LogisticRegression_f)
LogisticRegression_f.close()

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier,
                                  SGDClassifier_classifier,
                                  LogisticRegression_classifier)

def sentiment(text):
    feats = find_features(text)
    
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

#### This is part of the textblob package that generates a sentiment using textblobs algorithm

def sentiment_textblob(text):
    analyze = TextBlob(text)

    if analyze.polarity == 0.0:
        sentiment_value = "neutral"
        return sentiment_value, analyze.polarity
    elif analyze.polarity < 0.0:
        sentiment_value = "negative"
        return sentiment_value, analyze.polarity
    elif analyze.polarity > 0.0:
        sentiment_value = "positive"
        return sentiment_value, analyze.polarity

    #return sentiment_value, analyze.polarity


#### This is part of the nltk package that generate a sentiment using SentimentIntensityAnalyzer algorithm

def sentiment_nltk(text):
    sia = SentimentIntensityAnalyzer()

    if sia.polarity_scores(text)["compound"] == 0.0:
        sentiment_value = "neutral"
        return sentiment_value, sia.polarity_scores(text)["compound"]
    elif sia.polarity_scores(text)["compound"] < 0.0:
        sentiment_value = "negative"
        return sentiment_value, sia.polarity_scores(text)["compound"]
    elif sia.polarity_scores(text)["compound"] > 0.0:
        sentiment_value = "positive"
        return sentiment_value, sia.polarity_scores(text)["compound"]
