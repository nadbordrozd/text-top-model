import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer


class KerasExample(object):

    def __init__(self):
        self.vocab_size = None
        self.num_classes = None
        self.model = None

        self.max_words = 1000
        self.tokenizer = Tokenizer(num_words=self.max_words)

    def set_vocab_size(self, n):
        self.vocab_size = n

    def set_class_count(self, n):
        self.class_count = n

    def fit(self, X, y):
        if self.vocab_size is None or self.class_count is None:
            raise ValueError(
                "Must set vocab size and class count before training")

        X = self.tokenizer.sequences_to_matrix(X, mode='binary')
        y = keras.utils.to_categorical(y, self.class_count)

        batch_size = 32
        epochs = 5

        model = Sequential()
        model.add(Dense(512, input_shape=(self.max_words,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.class_count))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        self.history = model.fit(X, y,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 validation_split=0.1)
        self.model = model
        return self

    def predict_proba(self, X):
        X = self.tokenizer.sequences_to_matrix(X, mode='binary')
        return self.model.predict(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def __str__(self):
        return "KerasExample"
