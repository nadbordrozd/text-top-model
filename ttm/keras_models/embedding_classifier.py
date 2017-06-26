import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

from utils import get_embedding_matrix, get_embedding_dim


class EmbeddingClassifier(object):
    """superclass of all keras classifiers classifiers"""

    def __init__(self, epochs, max_seq_len, embedding_dim=30, embeddings_path=None,
                 optimizer='adam'):
        self.model = None
        self.vocab_size = None
        self.vocab = None
        self.num_classes = None
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.embeddings_path = embeddings_path
        if embeddings_path is not None:
            self.embedding_dim = get_embedding_dim(embeddings_path)
        self.optimizer = optimizer
        self.epochs = epochs

    def set_vocab(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def set_class_count(self, n):
        self.num_classes = n

    def build_embedding_layer(self):
        if self.embeddings_path is None:
            embedding_matrix = np.random.normal(size=(self.vocab_size, self.embedding_dim))
            trainable = True
        else:
            embedding_matrix = get_embedding_matrix(
                self.vocab, self.embeddings_path)
            trainable = False

        embedding_layer = Embedding(
            self.vocab_size,
            self.embedding_dim,
            weights=[embedding_matrix],
            input_length=self.max_seq_len,
            trainable=trainable)

        return embedding_layer

    def validate_params(self):
        if self.vocab_size is None or self.num_classes is None:
            raise ValueError(
                "Must set vocab size and class count before training")

    def transform_embedded_sequences(self, embedded_sequences):
        """this is the only method that most models will need to override. It takes sequence of
        embedded words and should return predictions.

        PLS OVERRIDE ME!
        """
        x = Flatten()(embedded_sequences)
        preds = Dense(self.num_classes, activation='softmax')(x)
        return preds

    def build_model(self):
        sequence_input = Input(shape=(self.max_seq_len,), dtype='int32')
        embedding_layer = self.build_embedding_layer()
        embedded_sequences = embedding_layer(sequence_input)
        predictions = self.transform_embedded_sequences(embedded_sequences)
        model = Model(sequence_input, predictions)
        return model

    def fit(self, X, y):
        self.validate_params()
        model = self.build_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer,
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

    def get_params(self):
        return {
            'epochs': self.epochs,
            'max_seq_len': self.max_seq_len,
            'embedding_dim': self.embedding_dim,
            'embeddings_path': self.embeddings_path,
            'optimizer': self.optimizer
        }

    def __str__(self):
        class_name = str(self.__class__).split('.')[-1]
        param_string = ", ".join('%s=%s' % (k, v) for k, v in self.get_params().items())
        return "%s(%s)" % (class_name, param_string)
