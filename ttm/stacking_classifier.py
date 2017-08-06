import numpy as np
from prepare_data import cache
from sklearn.model_selection import cross_val_predict


@cache
def cached_cv_predict(clf_class, params, X, y, cv, method):
    clf = clf_class(**params)
    return cross_val_predict(clf, X, y, cv=cv, method=method)


class StackingTextClassifier(object):
    def __init__(self, base_classifiers, stacker, folds=5, use_proba=True, vocab=None,
                 vocab_size=None, class_count=None, **kwargs):
        self.base_classifiers = []
        for clf, params in base_classifiers:
            params['vocab_size'] = vocab_size
            params['vocab'] = vocab
            params['class_count'] = class_count
            self.base_classifiers.append(clf(**params))

        stacker_class, stacker_params = stacker
        self.stacker = stacker_class(**stacker_params)
        self.folds = folds
        self.use_proba = use_proba

    def fit(self, X, y):
        n = len(y)
        method = 'predict_proba' if self.use_proba else 'predict'
        base_preds = [cached_cv_predict(
            clf.__class__, clf.get_params(), X, y, cv=self.folds, method=method)
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

    def __str__(self):
        base_clfs = ", ".join(str(x) for x in self.base_classifiers)

        return "StackingTextClassifier([%s], %s, folds=%s, use_proba=%s)" % (
            base_clfs, self.stacker, self.folds, self.use_proba)
