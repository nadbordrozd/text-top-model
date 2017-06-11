from sklearn_models import MultNB, BernNB, SVM
from keras_models import KerasExample
from benchmarks import benchmark


TRAIN_SET_PATH = '../data/r8-all-terms.txt'

models = [KerasExample(), MultNB(), BernNB(), SVM()]

if __name__ == '__main__':
    for model in models:
        print str(model), benchmark(model, TRAIN_SET_PATH)
