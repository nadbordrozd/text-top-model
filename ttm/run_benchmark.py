import numpy as np
import json

from sklearn_models import MultNB, BernNB, SVM
from keras_models.cnn import FCholletCNN
from keras_models.mlp import MLP
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
    (FCholletCNN, {'dropout_rate': 0.5, 'embedding_dim': 37, 'units': 400, 'epochs': 30}),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400, 'embeddings_path':
        '../data/glove.6B.100d.txt'}),
    (MLP, {'layers': 1, 'units': 360, 'dropout_rate': 0.87, 'epochs': 12, 'max_vocab_size': 22000}),
    (MLP, {'layers': 2, 'units': 180, 'dropout_rate': 0.6, 'epochs': 5, 'max_vocab_size': 22000}),
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
    (SVM, {'tfidf': False, 'kernel': 'linear', 'ngram_n': 2})
]

results_path = 'results.json'

if __name__ == '__main__':
    records = []
    for data_path in datasets:
        print
        print data_path

        for model_class, params in models:
            scores, times = benchmark(model_class, data_path, params, 10)
            model_str = str(model_class(**params))
            print '%.3f' % np.mean(scores), model_str
            for score, time in zip(scores, times):
                records.append({
                    'model': model_str,
                    'dataset': data_path,
                    'score': score,
                    'time': time
                })

    with open(results_path, 'wb') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
