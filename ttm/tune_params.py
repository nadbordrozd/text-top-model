from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pprint

from benchmarks import benchmark_with_early_stopping, cache
from keras_models.mlp import MLP
from keras_models.lstm import LSTMClassifier
from keras_models.blstm_2dcnn import BLSTM2DCNN
from keras_models.ykim_cnn import YKimCNN

def fix_ints(d):
    return {
        k: int(v) if (isinstance(v, float) and int(v) == v) else v
        for k, v in d.items()
    }


@cache
def hyperopt_me_like_one_of_your_french_girls(
        classifier, data_path, space, max_evals):
    def objective(args):
        best_loss, best_acc, best_epoch = benchmark_with_early_stopping(classifier, data_path,
                                                                        fix_ints(args))
        return {
            'loss': best_loss,
            'accuracy': best_acc,
            'epochs': best_epoch,
            'status': STATUS_OK
        }

    trials = Trials()
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials)

    return trials


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    DATA_PATH = '../data/polarity.txt'

    trials = hyperopt_me_like_one_of_your_french_girls(
        YKimCNN, DATA_PATH, {
            'units': hp.quniform('units', 10, 100, 10),
            'dropout_rates': [hp.uniform('dropout_1', 0.1, 0.9), hp.uniform('dropout_2', 0.1, 0.9)],
            'num_filters': hp.quniform('num_filters', 5, 100, 5),
            'filter_sizes': hp.choice('filter_sizes', [
                [3, 8],
                [3, 5],
                [3, 6],
                [3, 4, 5],
                [3, 5, 8],
                [3, 5, 7],
                [3, 4, 5, 6]
            ]),
            'embedding_dim': hp.quniform('embedding_dim', 5, 60, 5),
            'epochs': 200,
            'max_seq_len': 50
        }, max_evals=100)

    print '\n\nYKimCNN trainable embedding'
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        YKimCNN, DATA_PATH, {
            'units': hp.quniform('units', 10, 100, 10),
            'dropout_rates': [hp.uniform('dropout_1', 0.1, 0.9), hp.uniform('dropout_2', 0.1, 0.9)],
            'num_filters': hp.quniform('num_filters', 5, 100, 5),
            'filter_sizes': hp.choice('filter_sizes', [
                [3, 8],
                [3, 5],
                [3, 6],
                [3, 4, 5],
                [3, 5, 8],
                [3, 5, 7],
                [3, 4, 5, 6]
            ]),
            'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
            'epochs': 200,
            'max_seq_len': 50
        }, max_evals=100)

    print '\n\nYKimCNN glove embedding'
    pp.pprint(trials.best_trial)


    trials = hyperopt_me_like_one_of_your_french_girls(
        BLSTM2DCNN, DATA_PATH, {
            'units': hp.quniform('units', 8, 128, 4),
            'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.95),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.95),
            'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
            'bidirectional': True,
            'batch_size': 64
        }, max_evals=100)

    print '\n\nBLSTM with pretrained embedding'
    pp.pprint(trials.best_trial)


    trials = hyperopt_me_like_one_of_your_french_girls(
        BLSTM2DCNN, DATA_PATH, {
            'units': hp.quniform('units', 8, 256, 1),
            'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.9),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.9),
            'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embedding_dim': hp.quniform('embedding_dim', 2, 40, 1)
        }, max_evals=100)

    print '\n\nBLSTM2DCNN with trainable embedding'
    pp.pprint(trials.best_trial)


    trials = hyperopt_me_like_one_of_your_french_girls(
        MLP, DATA_PATH, {
            'layers': hp.quniform('layers', 1, 5, 1),
            'units': hp.quniform('units', 8, 2048, 1),
            'dropout_rate': hp.uniform('dropout_rate', 0.01, 0.99),
            'epochs': 200,
            'max_vocab_size': hp.quniform('max_vocab_size', 4000, 25000, 1000)
        }, max_evals=200)

    print '\n\nMLP'
    pp.pprint(trials.best_trial)


    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, DATA_PATH, {
            'layers': hp.quniform('layers', 1, 4, 1),
            'units': hp.quniform('units', 8, 256, 1),
            'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.9),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.9),
            'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embedding_dim': hp.quniform('embedding_dim', 2, 40, 1)
        }, max_evals=100)

    print '\n\nLSTM'
    pp.pprint(trials.best_trial)


    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, DATA_PATH, {
            'layers': hp.quniform('layers', 1, 3, 1),
            'units': hp.quniform('units', 8, 128, 1),
            'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.95),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.95),
            'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embedding_dim': hp.quniform('embedding_dim', 2, 40, 1),
            'bidirectional': True,
            'batch_size': 64
        }, max_evals=100)

    print '\n\nBLSTM'
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, DATA_PATH, {
            'layers': hp.quniform('layers', 1, 3, 1),
            'units': hp.quniform('units', 8, 128, 4),
            'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.95),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.95),
            'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
            'bidirectional': True,
            'batch_size': 64
        }, max_evals=100)

    print '\n\nBLSTM with pretrained embedding'
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, DATA_PATH, {
            'layers': hp.quniform('layers', 1, 3, 1),
            'units': hp.quniform('units', 8, 128, 4),
            'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.95),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.95),
            'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
            'batch_size': 64
        }, max_evals=100)

    print '\n\nLSTM with pretrained embedding'
    pp.pprint(trials.best_trial)
