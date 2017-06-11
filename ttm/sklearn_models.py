from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier




class MultNB(object):
    def __init__(self):
        self.clf = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
                             ("multinomial nb", MultinomialNB())])

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def __str__(self):
        return "MultinomialNB"


class BernNB(object):
    def __init__(self):
        self.clf = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
                             ("bernoulli nb", BernoulliNB())])

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def __str__(self):
        return "BernoulliNB"


class SVM(object):
    def __init__(self):
        self.clf = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
                             ("SVC", SVC())])

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def __str__(self):
        return "SVM"


class XGB(object):
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer=lambda x: x)
        self.clf = XGBClassifier()

    def set_vocab_size(self, n):
        self.vectorizer.fit([range(n)])
        return self

    def fit(self, X, y):
        self.clf.fit(self.vectorizer.transform(X), y)
        return self

    def predict(self, X):
        return self.clf.predict(self.vectorizer.transform(X))


    def __str__(self):
        return "XGBoost"



