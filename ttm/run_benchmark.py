from sklearn_models import MultNB, BernNB, SVM
from keras_models.cnn import FCholletCNN
from keras_models.mlp import MLP
from keras_models.lstm import LSTMClassifier
from tflearn_models import TFNN
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
    (LSTMClassifier, {'layers': 1, 'embedding_dim': 64, 'epochs': 15, 'batch_size': 256}),
    (LSTMClassifier, {'layers': 2, 'embedding_dim': 64, 'epochs': 15, 'batch_size': 128}),
    (LSTMClassifier, {'layers': 3, 'embedding_dim': 64, 'epochs': 15, 'batch_size': 128}),
    (FCholletCNN, {'epochs': 20, 'dropout_rate': 0, 'embedding_dim': 10}),
    (FCholletCNN, {'epochs': 20, 'dropout_rate': 0, 'embedding_dim': 20}),
    (FCholletCNN, {'epochs': 20, 'dropout_rate': 0, 'embedding_dim': 50}),
    (TFNN, {'layers': 1, 'units': 512, 'epochs': 20}),
    (TFNN, {'layers': 2, 'units': 512, 'epochs': 20}),
    (TFNN, {'layers': 4, 'units': 512, 'epochs': 20}),
    (MLP, {'layers': 1, 'dropout_rate': 0.2, 'epochs': 20}),
    (MLP, {'layers': 2, 'dropout_rate': 0.2, 'epochs': 20}),
    (MLP, {'layers': 3, 'dropout_rate': 0.2, 'epochs': 20}),
    (MultNB, {'tfidf': True}),
    (MultNB, {'tfidf': True, 'ngram_n': 2}),
    (MultNB, {'tfidf': True, 'ngram_n': 3}),
    (BernNB, {'tfidf': True}),
    (MultNB, {'tfidf': False}),
    (MultNB, {'tfidf': False, 'ngram_n': 2}),
    (BernNB, {'tfidf': False}),
    (SVM, {'tfidf': True, 'kernel': 'linear'}),
    (SVM, {'tfidf': True, 'kernel': 'linear', 'ngram_n': 2}),
    (SVM, {'tfidf': False, 'kernel': 'linear'}),
    (SVM, {'tfidf': False, 'kernel': 'linear', 'ngram_n': 2}),
]


if __name__ == '__main__':
    for data_path in datasets:
        print
        print data_path
        for model_class, params in models:
            score = benchmark(model_class, data_path, params)
            print "%.3f" % score, model_class(**params)
