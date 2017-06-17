import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer


class MLP(object):

    def __init__(self, layers=1, units=512, dropout_rate=0.5, epochs=5, batch_size=128):
        self.layers = layers
        self.units = units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_words = 1000
        self.tokenizer = Tokenizer(num_words=self.max_words)

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

        X = self.tokenizer.sequences_to_matrix(X, mode='binary')
        y = keras.utils.to_categorical(y, self.num_classes)

        model = Sequential()
        for i in range(self.layers):
            if i == 0:
                model.add(Dense(self.units, input_shape=(self.max_words,)))
            else:
                model.add(Dense(self.units))
            model.add(Activation('relu'))
            model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.history = model.fit(X, y,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 verbose=1)
        self.model = model
        return self

    def predict_proba(self, X):
        X = self.tokenizer.sequences_to_matrix(X, mode='binary')
        return self.model.predict(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def __str__(self):
        return "MLP(layers=%s, units=%s, dropout_rate=%s, epochs=%s, batch_size=%s)" % (
            self.layers, self.units, self.dropout_rate, self.epochs, self.batch_size)
