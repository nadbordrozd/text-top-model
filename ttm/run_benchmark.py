from sklearn_models import MultNB, BernNB, SVM
from keras_models import BasicNN
from benchmarks import benchmark


TRAIN_SET_PATH = '../data/r8-all-terms.txt'

models = [
    BasicNN,
    (BasicNN, {'layers': 2}),
    (BasicNN, {'layers': 3, 'units': 64}),
    MultNB,
    BernNB,
    SVM
]

if __name__ == '__main__':
    for model_class in models:
        params = {}
        if type(model_class) == tuple:
            model_class, params = model_class

        score = benchmark(model_class, TRAIN_SET_PATH, params)
        print "%.3f" % score, model_class(**params)
