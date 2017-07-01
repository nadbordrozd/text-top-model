import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from prepare_data import prepare_dataset, cache

@cache
def benchmark(model_class, data_path, model_params=None):
    """benchmarks a given model on a given dataset
    Instantiates the model with given parameters.
    :param model_class: class of the model to instantiate
    :param data_path: path to file with dataset
    :param model_params: optional dictionary with model parameters
    :return: accuracy score of the model on the dataset
    """
    if model_params is None:
        model_params = {}
    model = model_class(**model_params)
    X, y, vocab, label_encoder = prepare_dataset(data_path)
    vocab_size = len(vocab)
    class_count = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

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

    preds = model.fit(X_train, y_train).predict(X_test)
    return accuracy_score(preds, y_test)


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
    vocab_size = len(vocab)
    class_count = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

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

    model.fit(X_train, y_train, validation_data=[X_test, y_test])
    best_loss = np.min(model.history.history['val_loss'])
    best_acc = np.max(model.history.history['val_acc'])
    best_epoch = np.argmin(model.history.history['val_loss']) + 1

    return best_loss, best_acc, best_epoch
