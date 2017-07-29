import numpy as np
from time import time

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from prepare_data import prepare_dataset, cache


def set_metadata(model, vocab, class_count):
    vocab_size = len(vocab)

    try:
        model.set_vocab(vocab)
    except AttributeError:
        pass

    try:
        model.set_vocab_size(vocab_size)
    except AttributeError:
        pass

    try:
        model.set_class_count(class_count)
    except AttributeError:
        pass


@cache
def benchmark(model_class, data_path, model_params=None, iters=1):
    """benchmarks a given model on a given dataset
    Instantiates the model with given parameters.
    :param model_class: class of the model to instantiate
    :param data_path: path to file with dataset
    :param model_params: optional dictionary with model parameters
    :param iters: how many times to benchmark
    :param return_time: if true, returns list of running times in addition to scores
    :return: tuple (accuracy scores, running times)
    """
    if model_params is None:
        model_params = {}

    X, y, vocab, label_encoder = prepare_dataset(data_path)
    class_count = len(label_encoder.classes_)

    scores = []
    times = []
    for i in range(iters):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model = model_class(**model_params)
        set_metadata(model, vocab, class_count)
        start = time()
        preds = model.fit(X_train, y_train).predict(X_test)
        end = time()
        scores.append(accuracy_score(preds, y_test))
        times.append(end - start)
    return scores, times


@cache
def benchmark_with_early_stopping(model_class, data_path, model_params=None):
    """same as benchmark but fits with validation data to allow the model to do early stopping
    Works with all models from keras_models
    :param model_class: class of the model to instantiate, must have fit(X, y, validation_data)
        method and 'history' attribute
    :param data_path: path to file with dataset
    :param model_params: optional dictionary with model parameters
    :return: best_loss, best_score, best_epoch
    """
    if model_params is None:
        model_params = {}
    model = model_class(**model_params)
    X, y, vocab, label_encoder = prepare_dataset(data_path)
    class_count = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    set_metadata(model, vocab, class_count)

    model.fit(X_train, y_train, validation_data=[X_test, y_test])
    best_loss = np.min(model.history.history['val_loss'])
    best_acc = np.max(model.history.history['val_acc'])
    best_epoch = np.argmin(model.history.history['val_loss']) + 1

    print model, "acc", best_acc, "loss",  best_loss, "epochs", best_epoch
    return best_loss, best_acc, best_epoch
