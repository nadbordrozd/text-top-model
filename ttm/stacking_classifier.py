import numpy as np
from sklearn.model_selection import cross_val_predict


class StackingTextClassifier(object):
    def __init__(self, base_classifiers, stacker, folds=5, use_proba=True):
        self.base_classifiers = base_classifiers
        self.stacker = stacker
        self.folds = folds
        self.use_proba = use_proba

    def fit(self, X, y):
        n = len(y)
        method = 'predict_proba' if self.use_proba else 'predict'
        base_preds = [cross_val_predict(clf, X, y, cv=self.folds, method=method)
                      for clf in self.base_classifiers]
        if not self.use_proba:
            base_preds = [x.reshape(n, 1) for x in base_preds]

        self.stacker.fit(np.hstack(base_preds), y)
        for clf in self.base_classifiers:
            clf.fit(X, y)
        return self

    def predict(self, X):
        n = len(X)
        if self.use_proba:
            base_preds = [clf.predict_proba(X) for clf in self.base_classifiers]
        else:
            base_preds = [clf.predict(X).reshape(n, 1) for clf in self.base_classifiers]
        return self.stacker.predict(np.hstack(base_preds))

    def predict_proba(self, X):
        n = len(X)
        if self.use_proba:
            base_preds = [clf.predict_proba(X) for clf in self.base_classifiers]
        else:
            base_preds = [clf.predict(X).reshape(n, 1) for clf in self.base_classifiers]
        return self.stacker.predict_proba(np.hstack(base_preds))

    def set_vocab(self, vocab):
        for clf in self.base_classifiers:
            try:
                clf.set_vocab(vocab)
            except AttributeError:
                pass

    def set_class_count(self, n):
        for clf in self.base_classifiers:
            try:
                clf.set_class_count(n)
            except AttributeError:
                pass

    def __str__(self):
        base_clfs = ", ".join(str(x) for x in self.base_classifiers)

        return "StackingTextClassifier([%s], %s, folds=%s, use_proba=%s)" % (
            base_clfs, self.stacker, self.folds, self.use_proba)


def stacking_text_classifier(base_classifiers, stacker, folds=5, use_proba=True):
    """Alternative way to instantiate StackingTextClassifier.
     StackingTextClassifier.__init__ takes *instances* of base classifiers and stacker, which
     interferes with caching (because the instances are not picklable). This function is a wrapper
     around StackingTextClassifier.__init__ that accepts model *classes* instead. For example,
     __init__ would accept argument stacker=MultinomialNaiveBayes(alpha=10)
    this function instead accepts
    stacker=(MultinomialNaiveBayes, {'alpha': 10})
    This latter way is joblib-caching-friendly

    :param base_classifiers: list of tuples (classifier class, dictionary of params)
    :param stacker: tuple (classifier class, dictionary of params)
    :param folds: number of folds
    :param use_proba: whether to use predict_proba with base classifiers
    :return: instance of StackingTextClassifier
    """
    inst_base_clf = [model(**params) for model, params in base_classifiers]
    stacker_class, stacker_params = stacker
    inst_stacker = stacker_class(**stacker_params)
    return StackingTextClassifier(inst_base_clf, inst_stacker, folds, use_proba)
