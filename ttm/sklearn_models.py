from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier


class SklearnClassifierWrapper(object):
    def __init__(self, model, tfidf=False):
        """
        Classifier made up of a pipeline with a count vectorizer + given model
        :param model: a sklearn-like classifier (with fit, predict and predict_proba)
        :param tfidf: if True wil use TfidfVectorizer, otherwise CountVectorizer; defaults to False
        """
        if tfidf:
            vectorizer_step = ('tfidf_vectorizer', TfidfVectorizer(analyzer=lambda x: x))
        else:
            vectorizer_step = ('count_vectorizer', CountVectorizer(analyzer=lambda x: x))

        self.clf = Pipeline([
            vectorizer_step,
            ('model', model)])
        self.name = "SklearnClassifierWrapper(tfidf=%s)" % tfidf

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)

    def __str__(self):
        return self.name


class MultNB(SklearnClassifierWrapper):
    def __init__(self, tfidf=False):
        super(MultNB, self).__init__(MultinomialNB(), tfidf)
        self.name = "MultinomialNB(tfidf=%s)" % tfidf


class BernNB(SklearnClassifierWrapper):
    def __init__(self, tfidf=False):
        super(BernNB, self).__init__(BernoulliNB(), tfidf)
        self.name = "BernoulliNB(tfidf=%s)" % tfidf


class SVM(SklearnClassifierWrapper):
    def __init__(self, tfidf=False, kernel='linear'):
        super(SVM, self).__init__(SVC(kernel=kernel), tfidf)
        self.name = "SVC(tfidf=%s, kernel=%s)" % (tfidf, kernel)


class XGB(SklearnClassifierWrapper):
    def __init__(self, tfidf=False):
        super(XGB, self).__init__(XGBClassifier(), tfidf)
        self.name = "XGBClassifier(tfidf=%s)" % tfidf

