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
