import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


class FCholletCNN(object):
    """Based on
    https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    except with trainable embeddings instead of pretrained from GloVe"""

    def __init__(self, epochs=20, embedding_dim=20, units=128, dropout_rate=0):
        self.model = None
        self.vocab_size = None
        self.num_classes = None
        self.embedding_dim = embedding_dim
        self.max_seq_len = 1000

        self.units = units
        self.dropout_rate = dropout_rate
        self.epochs = epochs

    def set_vocab_size(self, n):
        self.vocab_size = n

    def set_class_count(self, n):
        self.num_classes = n

    def fit(self, X, y):
        if self.vocab_size is None or self.num_classes is None:
            raise ValueError(
                "Must set vocab size and class count before training")

        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        embedding_layer = Embedding(
            self.vocab_size,
            self.embedding_dim,
            weights=[embedding_matrix],
            input_length=self.max_seq_len,
            trainable=True)

        sequence_input = Input(shape=(self.max_seq_len,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(self.units, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Conv1D(self.units, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Conv1D(self.units, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Flatten()(x)
        x = Dense(self.units, activation='relu')(x)
        preds = Dense(self.num_classes, activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                      metrics=['acc'])

        padded_X = pad_sequences(X, self.max_seq_len)
        one_hot_y = to_categorical(y, num_classes=self.num_classes)
        model.fit(padded_X, one_hot_y, epochs=self.epochs)
        self.model = model
        return self

    def predict_proba(self, X):
        X = pad_sequences(X, self.max_seq_len)
        return self.model.predict(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def __str__(self):
        return "FCholletCNN(units=%s, dropout_rate=%s, epochs=%s, embedding_dim=%s)" % (
            self.units, self.dropout_rate, self.epochs, self.embedding_dim)
