from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from prepare_data import prepare_dataset, cache



def benchmark(model, data_path):
    X, y, word_encoder, label_encoder = prepare_dataset(data_path)
    vocab_size = len(word_encoder.classes_)
    class_count = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

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
