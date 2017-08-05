import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping

from utils import get_embedding_matrix, get_embedding_dim


class KerasTextClassifier(object):
    """superclass of all keras classifiers classifiers"""

    def __init__(
            self,
            max_seq_len=100,
            embedding_dim=20,
            embeddings_path=None,
            optimizer='adam',
            batch_size=32,
            epochs=10,
            vocab=None,
            vocab_size=None,
            class_count=None,
            **kwargs):

        self.vocab_size = vocab_size
        self.vocab = vocab
        self.class_count = class_count
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.embeddings_path = embeddings_path
        if embeddings_path is not None:
            self.embedding_dim = get_embedding_dim(embeddings_path)
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = 3
        self.history = None
        self.model = None

        self.params = {
            'epochs': self.epochs,
            'max_seq_len': self.max_seq_len,
            'embedding_dim': self.embedding_dim,
            'embeddings_path': self.embeddings_path,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size,
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'class_count': self.class_count
        }

    def build_embedding_layer(self):
        if self.embeddings_path is None:
            embedding_matrix = np.random.normal(
                size=(self.vocab_size, self.embedding_dim))
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
        if self.vocab_size is None or self.class_count is None:
            raise ValueError(
                "Must set vocab size and class count before training")

    def transform_embedded_sequences(self, embedded_sequences):
        """this is the only method that most models will need to override. It takes sequence of
        embedded words and should return predictions.

        PLS OVERRIDE ME!
        """
        x = Flatten()(embedded_sequences)
        preds = Dense(self.class_count, activation='softmax')(x)
        return preds

    def build_model(self):
        sequence_input = Input(shape=(self.max_seq_len,), dtype='int32')
        embedding_layer = self.build_embedding_layer()
        embedded_sequences = embedding_layer(sequence_input)
        predictions = self.transform_embedded_sequences(embedded_sequences)
        model = Model(sequence_input, predictions)
        return model

    def fit(self, X, y, validation_data=None):
        self.validate_params()
        model = self.build_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['acc'])

        padded_X = pad_sequences(X, self.max_seq_len)
        one_hot_y = to_categorical(y, num_classes=self.class_count)

        if validation_data is not None:
            v_X, v_y = validation_data
            v_X = pad_sequences(v_X, self.max_seq_len)
            v_y = to_categorical(v_y, self.class_count)

            early_stopping = EarlyStopping(
                monitor='val_loss', patience=self.patience)
            self.history = model.fit(
                padded_X, one_hot_y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=[v_X, v_y],
                callbacks=[early_stopping])
        else:
            self.history = model.fit(
                padded_X,
                one_hot_y,
                batch_size=self.batch_size,
                epochs=self.epochs)
        self.model = model
        return self

    def predict_proba(self, X):
        X = pad_sequences(X, self.max_seq_len)
        return self.model.predict(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def get_params(self, deep=None):
        return self.params

    def __str__(self):
        class_name = str(self.__class__).split('.')[-1][:-2]
        param_string = ", ".join(
            '%s=%s' % (k, v)
            for k, v in self.get_params().items()
            if k not in ['vocab', 'vocab_size', 'class_count']
        )
        return "%s(%s)" % (class_name, param_string)
