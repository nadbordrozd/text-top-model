from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pprint

from benchmarks import benchmark_with_early_stopping, cache
from keras_models.mlp import MLP
from keras_models.lstm import LSTMClassifier
from keras_models.blstm_2dcnn import BLSTM2DCNN
from keras_models.cnn import FCholletCNN


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
    DATA_PATH = '../data/20ng-all-terms.txt'

    trials = hyperopt_me_like_one_of_your_french_girls(
        MLP, DATA_PATH, {
            'layers': hp.quniform('layers', 1, 5, 1),
            'units': hp.quniform('units', 8, 2048, 1),
            'dropout_rate': hp.uniform('dropout_rate', 0.01, 0.99),
            'epochs': 200,
            'max_vocab_size': hp.quniform('max_vocab_size', 4000, 22000, 1000)
        }, max_evals=300)

    print '\n\nMLP'
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        FCholletCNN, DATA_PATH, {
            'units': hp.quniform('units', 16, 512, 16),
            'dropout_rate': hp.uniform('dropout_rate', 0, 0.9),
            'epochs': 200,
            'embedding_dim': hp.quniform('embedding_dim', 2, 40, 1)
        }, max_evals=50)

    print '\n\nFChollet'
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        FCholletCNN, DATA_PATH, {
            'units': hp.quniform('units', 16, 512, 16),
            'dropout_rate': hp.uniform('dropout_rate', 0, 0.9),
            'epochs': 200,
            'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
        }, max_evals=50)

    print '\n\nFChollet with pretrained embeddings'
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, DATA_PATH, {
            'layers': hp.quniform('layers', 1, 4, 1),
            'units': hp.quniform('units', 8, 128, 1),
            'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.7),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0.2, 0.7),
            'epochs': 30,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embedding_dim': hp.quniform('embedding_dim', 2, 40, 1)
        }, max_evals=5)

    print '\n\nLSTM'
    pp.pprint(trials.best_trial)


    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, DATA_PATH, {
            'layers': hp.quniform('layers', 1, 3, 1),
            'units': hp.quniform('units', 8, 64, 1),
            'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.7),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0.2, 0.7),
            'epochs': 30,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embedding_dim': hp.quniform('embedding_dim', 2, 40, 1),
            'bidirectional': True,
            'batch_size': 64
        }, max_evals=10)

    print '\n\nBLSTM'
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, DATA_PATH, {
            'layers': hp.quniform('layers', 1, 3, 1),
            'units': hp.quniform('units', 8, 64, 1),
            'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.7),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0.2, 0.7),
            'epochs': 30,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
            'bidirectional': True,
            'batch_size': 64
        }, max_evals=5)

    print '\n\nBLSTM with pretrained embedding'
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, DATA_PATH, {
            'layers': hp.quniform('layers', 1, 3, 1),
            'units': hp.quniform('units', 8, 64, 1),
            'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.7),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0.2, 0.7),
            'epochs': 30,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
            'batch_size': 64
        }, max_evals=5)

    print '\n\nLSTM with pretrained embedding'
    pp.pprint(trials.best_trial)
