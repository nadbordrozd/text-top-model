from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pprint

from benchmarks import benchmark, cache
from keras_models.mlp import MLP


def fix_ints(d):
    return {
        k: int(v) if (isinstance(v, float) and int(v) == v) else v
        for k, v in d.items()
    }


@cache
def hyperopt_me_like_one_of_your_french_girls(
        classifier, data_path, space, max_evals):
    def objective(args):
        return {
            'loss': benchmark(classifier, data_path, fix_ints(args)),
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
            'epochs': hp.quniform('epochs', 5, 150, 1),
            'max_vocab_size': hp.quniform('max_vocab_size', 4000, 22000, 1000)
        }, max_evals=200)

    print 'MLP'
    pp.pprint(trials.best_trial)
