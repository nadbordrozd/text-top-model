import numpy as np
import pandas as pd

from sklearn_models import MultNB, BernNB, SVM
from keras_models.fchollet_cnn import FCholletCNN
from keras_models.mlp import MLP
from benchmarks import benchmark

datasets = [
    '20ng-all-terms.txt',
    '20ng-no-short.txt',
    '20ng-no-stop.txt',
    '20ng-stemmed.txt',
    'r52-all-terms.txt',
    'r52-no-short.txt',
    'r52-no-stop.txt',
    'r52-stemmed.txt',
    'r8-all-terms.txt',
    'r8-no-short.txt',
    'r8-no-stop.txt',
    'r8-stemmed.txt',
    'webkb-stemmed.txt'
]

models = [
    (FCholletCNN, {'dropout_rate': 0.5, 'embedding_dim': 37, 'units': 400, 'epochs': 30}, "CNN 37D"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path':'../data/glove.6B/glove.6B.100d.txt'}, "CNN GloVe"),
    (MLP, {'layers': 1, 'units': 360, 'dropout_rate': 0.87, 'epochs': 12, 'max_vocab_size': 22000}, "MLP 1x360"),
    (MLP, {'layers': 2, 'units': 180, 'dropout_rate': 0.6, 'epochs': 5, 'max_vocab_size': 22000}, "MLP 2x180"),
    (MLP, {'layers': 3, 'dropout_rate': 0.2, 'epochs': 20}, "MLP 3x512"),
    (MultNB, {'tfidf': True}, "MNB tfidf"),
    (MultNB, {'tfidf': True, 'ngram_n': 2}, "MNB tfidf 2-gr"),
    (MultNB, {'tfidf': True, 'ngram_n': 3}, "MNB tfidf 3-gr"),
    (BernNB, {'tfidf': True}, "BNB tfidf"),
    (MultNB, {'tfidf': False}, "MNB"),
    (MultNB, {'tfidf': False, 'ngram_n': 2}, "MNB 2-gr"),
    (BernNB, {'tfidf': False}, "BNB"),
    (SVM, {'tfidf': True, 'kernel': 'linear'}, "SVM tfidf"),
    (SVM, {'tfidf': True, 'kernel': 'linear', 'ngram_n': 2}, "SVM tfidf 2-gr"),
    (SVM, {'tfidf': False, 'kernel': 'linear'}, "SVM"),
    (SVM, {'tfidf': False, 'kernel': 'linear', 'ngram_n': 2}, "SVM 2-gr")
]


results_path = 'document_results.csv'

if __name__ == '__main__':
    records = []
    for data_path in datasets:
        print
        print data_path

        for model_class, params, model_name in models:
            scores, times = benchmark(model_class, data_path, params, 10)
            model_str = str(model_class(**params))
            print '%.3f' % np.mean(scores), model_str
            for score, time in zip(scores, times):
                records.append({
                    'model': model_str,
                    'dataset_name': data_path.split("/")[-1],
                    'accuracy': score,
                    'time': time,
                    'model_name': model_name
                })

    pd.DataFrame(records).to_csv(results_path, index=False)
