import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

from utils import get_embedding_matrix


class FCholletCNN(object):
    """Based on
    https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    except with trainable embeddings instead of pretrained from GloVe"""

    def __init__(
            self,
            epochs=20,
            embedding_dim=20,
            embeddings_path=None,
            units=128,
            max_seq_len=1000,
            dropout_rate=0):

        self.model = None
        self.vocab_size = None
        self.vocab = None
        self.num_classes = None
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.embeddings_path = embeddings_path
        if embeddings_path is not None:
            self.embedding_dim = None

        self.units = units
        self.dropout_rate = dropout_rate
        self.epochs = epochs

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def set_vocab_size(self, n):
        self.vocab_size = n

    def set_class_count(self, n):
        self.num_classes = n

    def fit(self, X, y):
        if self.vocab_size is None or self.num_classes is None:
            raise ValueError(
                "Must set vocab size and class count before training")

        if self.embeddings_path is None:
            embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
            trainable = True
        else:
            embedding_matrix = get_embedding_matrix(
                self.vocab, self.embeddings_path)
            trainable = False
            _, self.embedding_dim = embedding_matrix.shape

        embedding_layer = Embedding(
            self.vocab_size,
            self.embedding_dim,
            weights=[embedding_matrix],
            input_length=self.max_seq_len,
            trainable=trainable)

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
        if self.embeddings_path is None:
            return "FCholletCNN(units=%s, dropout_rate=%s, max_seq_len=%s, epochs=%s, " \
                   "embedding_dim=%s)" % (
                       self.units, self.dropout_rate, self.max_seq_len, self.epochs, self.embedding_dim)
        else:
            return "FCholletCNN(units=%s, dropout_rate=%s, max_seq_len=%s, epochs=%s, " \
                   "embeddings_path=%s)" % (
                       self.units, self.dropout_rate, self.max_seq_len, self.epochs,
                       self.embeddings_path)
