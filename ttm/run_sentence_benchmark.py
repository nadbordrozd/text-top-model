import numpy as np
import pandas as pd

from sklearn_models import MultNB, BernNB, SVM
from keras_models.blstm_2dcnn import BLSTM2DCNN
from keras_models.lstm import LSTMClassifier
from keras_models.ykim_cnn import YKimCNN
from keras_models.mlp import MLP
from benchmarks import benchmark

datasets = [
    '../data/polarity.txt',
    '../data/subjectivity_10k.txt'
]


models = [
    (MLP, {'layers': 1, 'units': 360, 'dropout_rate': 0.87, 'epochs': 12, 'max_vocab_size': 22000}),
    (MLP, {'layers': 2, 'units': 180, 'dropout_rate': 0.6, 'epochs': 5, 'max_vocab_size': 22000}),
    (MLP, {'layers': 3, 'dropout_rate': 0.2, 'epochs': 20}),
    (LSTMClassifier, {
        'max_seq_len': 50,
        'layers': 3,
        'dropout_rate': 0.45,
        'rec_dropout_rate': 0.35,
        'optimizer': 'adam',
        'embedding_dim': 24,
        'epochs': 18,
        'bidirectional': False,
        'units': 250
    }),
    (LSTMClassifier, {
        'max_seq_len': 50,
        'layers': 2,
        'dropout_rate': 0.45,
        'rec_dropout_rate': 0.4,
        'optimizer': 'rmsprop',
        'embedding_dim': 12,
        'epochs': 60,
        'bidirectional': False,
        'units': 80
    }),
    (LSTMClassifier, {
        'max_seq_len': 50,
        'layers': 2,
        'dropout_rate': 0.25,
        'rec_dropout_rate': 0.5,
        'optimizer': 'rmsprop',
        'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
        'epochs': 42,
        'bidirectional': True,
        'units': 16}),
    (LSTMClassifier, {
        'max_seq_len': 50,
        'layers': 2,
        'dropout_rate': 0.25,
        'rec_dropout_rate': 0.5,
        'optimizer': 'rmsprop',
        'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
        'epochs': 42,
        'bidirectional': False,
        'units': 32
    }),
    (YKimCNN, {
        'max_seq_len': 50,
        'filter_sizes': (3, 5, 7),
        'num_filters': 5,
        'embedding_dim': 45,
        'dropout_rates': (0.64, 0.47),
        'units': 40,
        'epochs': 53,
        'batch_size': 128
    }),
    (YKimCNN, {
        'max_seq_len': 50,
        'filter_sizes': (3, 5),
        'num_filters': 75,
        'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
        'dropout_rates': (0.2, 0.8),
        'units': 50,
        'epochs': 33,
        'batch_size': 128
    }),
    (BLSTM2DCNN, {
        'max_seq_len': 50,
        'dropout_rate': 0.4,
        'rec_dropout_rate': 0.88,
        'optimizer': 'rmsprop',
        'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
        'units': 8,
        'conv_filters': 32,
        'epochs': 31,
        'batch_size': 64
    }),
    (BLSTM2DCNN, {
        'max_seq_len': 50,
        'dropout_rate': 0.4,
        'rec_dropout_rate': 0.75,
        'optimizer': 'adam',
        'embedding_dim': 15,
        'units': 162,
        'conv_filters': 32,
        'epochs': 26,
        'batch_size': 128
    }),
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

results_path = 'sentence_results.csv'

if __name__ == '__main__':
    records = []
    for data_path in datasets:
        print
        print data_path

        for model_class, params in models:
            scores, times = benchmark(model_class, data_path, params, 1)
            model_str = str(model_class(**params))
            print '%.3f' % np.mean(scores), model_str
            for score, time in zip(scores, times):
                records.append({
                    'model': model_str,
                    'dataset': data_path,
                    'score': score,
                    'time': time
                })

    pd.DataFrame(records).to_csv(results_path)
