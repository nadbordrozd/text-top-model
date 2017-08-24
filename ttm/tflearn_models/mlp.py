import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression

from keras.preprocessing.text import Tokenizer
# from sklearn.feature_extraction.text import CountVectorizer

class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, patience):
        self.patience = patience
        self.loss_value = None
        self.impatience = 0


    def on_epoch_end(self, training_state):
        # Apparently this can happen.
        if training_state.loss_value is None: return

        # initialise loss value
        if not self.loss_value:
            self.loss_value = training_state.loss_value

        if self.impatience >= self.patience:
            raise StopIteration

        # if loss does not decrease for patience >>consecutive<< epochs it
        # will raise StopIteration
        if training_state.loss_value >= self.loss_value:
            self.impatience+=1
        else:
            self.impatience = 0

        self.loss_value = training_state.loss_value


class MLP(object):

    def __init__(
            self,
            layers=1,
            units=512,
            dropout_rate=0.5,
            max_vocab_size=10000,
            epochs=5,
            batch_size=128,
            vocab_size=None,
            class_count=None,
            **kwargs):
        self.layers = layers
        self.units = units
        self.dropout_rate = 1-dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_classes = class_count
        self.max_vocab_size = max_vocab_size
        self.vocab_size = min(vocab_size, max_vocab_size)
        self.tokenizer = Tokenizer(num_words=self.vocab_size)
        # self.tokenizer = CountVectorizer(
        #     max_features=self.vocab_size,
        #     analyzer=lambda x: x,
        #     binary=True)

        self.model = None
        self.patience = 3
        self.history = None
        self.params = {
            'layers': layers,
            'units': units,
            'dropout_rate': dropout_rate,
            'max_vocab_size': max_vocab_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'vocab_size': vocab_size,
            'class_count': class_count
        }

    def fit(self, X, y, validation_data=None):
        if self.vocab_size is None or self.num_classes is None:
            raise ValueError(
                "Must set vocab size and class count before training")

        X = self.tokenizer.sequences_to_matrix(X, mode='binary')
        # X = np.asarray(self.tokenizer.fit_transform(X).todense())
        y = tflearn.data_utils.to_categorical(y, nb_classes=self.num_classes)

        net = tflearn.input_data(shape=[None, self.vocab_size], name='input')
        for _ in range(self.layers):
                net = fully_connected(net, self.units, activation='relu')
                net = dropout(net, self.dropout_rate)

        net = fully_connected(net, self.num_classes, activation='softmax')
        net = regression(net, optimizer='adam', loss='categorical_crossentropy', name='target')
        model = tflearn.DNN(net, tensorboard_verbose=0)

        if validation_data is not None:
            v_X, v_y = validation_data
            v_X = self.tokenizer.sequences_to_matrix(v_X, mode='binary')
            # v_X = np.asarray(self.tokenizer.fit_transform(v_X).todense())
            v_y = tflearn.data_utils.to_categorical(v_y, nb_classes=self.num_classes)

            early_stopping = EarlyStoppingCallback(patience=self.patience)
            history = model.fit(
                X, y,
                batch_size=self.batch_size,
                n_epoch=self.epochs,
                validation_set=(v_X, v_y),
                show_metric=True,
                callbacks=early_stopping)
        else:
            history = model.fit(
                X, y,
                batch_size=self.batch_size,
                n_epoch=self.epochs,
                show_metric=True
                )

        self.model = model
        return self

    def predict_proba(self, X):
        X = self.tokenizer.sequences_to_matrix(X, mode='binary')
        # X = np.asarray(self.tokenizer.fit_transform(X).todense())
        return self.model.predict(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def get_params(self, deep=None):
        return self.params

    def __str__(self):
        return "MLP(layers=%s, units=%s, dropout_rate=%s, max_vocab_size=%s, epochs=%s, " \
               "batch_size=%s)" % (
                   self.layers, self.units, self.dropout_rate, self.max_vocab_size, self.epochs,
                   self.batch_size)
