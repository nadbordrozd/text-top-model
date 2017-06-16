import tensorflow as tf
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical
# from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from prepare_data import prepare_dataset, cache


class TFNN(object):

    def __init__(self, layers=1, units=512, dropout_rate=0.5, epochs=5, batch_size=128):
        self.layers = layers
        self.units = units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_words = 1000
        self.tokenizer = Tokenizer(num_words=self.max_words)
        # self.tokenizer = CountVectorizer(max_features=self.max_words, analyzer=lambda x: x)

        self.vocab_size = None
        self.num_classes = None
        self.model = None

    def set_vocab_size(self, n):
        self.vocab_size = n

    def set_class_count(self, n):
        self.num_classes = n

    def fit(self, X, y):
        if self.vocab_size is None or self.num_classes is None:
            raise ValueError(
                "Must set vocab size and class count before training")

        # X = self.tokenizer.fit_transform(X).todense()
        X = self.tokenizer.sequences_to_matrix(X, mode='binary')
        y = to_categorical(y, self.num_classes)

        tf.reset_default_graph()
        net = input_data([None, self.max_words], name='input')
        for _ in range(self.layers):
            net = fully_connected(net, self.units, activation='relu')
            net = dropout(net, 0.5)
        net = fully_connected(net, self.num_classes, activation='softmax')

        net = regression(net, optimizer='adam', loss='categorical_crossentropy', name='target')
        model = tflearn.DNN(net, tensorboard_verbose=0)

        self.history = model.fit({'input': X}, {'target': y},
                                batch_size=self.batch_size,
                                n_epoch=self.epochs,
                                show_metric=True
                                )
        self.model = model
        return self

    def predict_proba(self, X):
        # X = self.tokenizer.fit_transform(X).todense()
        X = self.tokenizer.sequences_to_matrix(X, mode='binary')
        return np.array(self.model.predict(X))

    def predict(self, X):
        return np.array(self.predict_proba(X)).argmax(axis=1)

    def __str__(self):
        return "TFNN(layers=%s, units=%s, dropout_rate=%s, epochs=%s, batch_size=%s)" % (
            self.layers, self.units, self.dropout_rate, self.epochs, self.batch_size)


if __name__ == '__main__':

    data_path = '../data/r8-all-terms.txt'
    X, y, word_encoder, label_encoder = prepare_dataset(data_path)
    vocab_size = len(word_encoder.classes_)
    class_count = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model = TFNN()
    model.set_vocab_size(vocab_size)
    model.set_class_count(class_count)

    preds = model.fit(X_train, y_train).predict(X_test)
    print accuracy_score(preds, y_test)
