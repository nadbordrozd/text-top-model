from sklearn_models import MultNB, BernNB, SVM
from keras_models import BasicNN
from benchmarks import benchmark


datasets = [
    '../data/r8-all-terms.txt',
    '../data/r52-all-terms.txt',
    '../data/20ng-all-terms.txt',
    '../data/r8-no-short.txt',
    '../data/r52-no-short.txt',
    '../data/20ng-no-short.txt',
    '../data/r8-no-stop.txt',
    '../data/r52-no-stop.txt',
    '../data/20ng-no-stop.txt',
    '../data/r8-stemmed.txt',
    '../data/r52-stemmed.txt',
    '../data/20ng-stemmed.txt',
    '../data/webkb-stemmed.txt'
]

models = [
    BasicNN,
    (BasicNN, {'layers': 2}),
    (BasicNN, {'layers': 3, 'units': 64}),
    MultNB,
    BernNB,
    SVM,
    (SVM, {'kernel': 'rbf'}),
    (MultNB, {'tfidf': True}),
    (BernNB, {'tfidf': True}),
    (SVM, {'tfidf': True}),
    (SVM, {'tfidf': True, 'kernel': 'rbf'})
]

if __name__ == '__main__':
    for data_path in datasets:
        print
        print data_path
        for model_class in models:
            params = {}
            if type(model_class) == tuple:
                model_class, params = model_class

            score = benchmark(model_class, data_path, params)
            print "%.3f" % score, model_class(**params)
