import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping


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
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_classes = class_count
        self.max_vocab_size = max_vocab_size
        self.vocab_size = min(vocab_size, max_vocab_size)
        self.tokenizer = Tokenizer(num_words=self.vocab_size)

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
        y = keras.utils.to_categorical(y, self.num_classes)

        model = Sequential()
        for i in range(self.layers):
            if i == 0:
                model.add(Dense(self.units, input_shape=(self.vocab_size,)))
            else:
                model.add(Dense(self.units))
            model.add(Activation('relu'))
            model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if validation_data is not None:
            v_X, v_y = validation_data
            v_X = self.tokenizer.sequences_to_matrix(v_X, mode='binary')
            v_y = keras.utils.to_categorical(v_y, self.num_classes)

            early_stopping = EarlyStopping(
                monitor='val_loss', patience=self.patience)
            self.history = model.fit(
                X, y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=1,
                validation_data=[v_X, v_y],
                callbacks=[early_stopping])
        else:
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

    def get_params(self, deep=None):
        return self.params

    def __str__(self):
        return "MLP(layers=%s, units=%s, dropout_rate=%s, max_vocab_size=%s, epochs=%s, " \
               "batch_size=%s)" % (
                   self.layers, self.units, self.dropout_rate, self.max_vocab_size, self.epochs,
                   self.batch_size)
