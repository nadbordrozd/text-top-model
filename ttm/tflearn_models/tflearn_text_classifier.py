import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers import embedding
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

from utils import get_embedding_matrix, get_embedding_dim


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


class TFlearnTextClassifier(object):
    """superclass of all TF classifiers"""

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
            trainable = True
        else:
            trainable = False

        embedding_layer = input_data(shape=[None,self.max_seq_len], name='input')
        embedding_layer = embedding(
            embedding_layer,
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            trainable=trainable,
            name="EmbeddingLayer")

        return trainable, embedding_layer

    def validate_params(self):
        if self.vocab_size is None or self.class_count is None:
            raise ValueError(
                "Must set vocab size and class count before training")


    def transform_embedded_sequences(self, embedded_sequences):
        """this is the only method that most models will need to override. It takes sequence of
        embedded words and should return predictions.

        PLS OVERRIDE ME!
        """
        preds = fully_connected(embedded_sequences, self.class_count, activation='softmax')
        return preds


    def build_model(self):

        trainable, net = self.build_embedding_layer()
        net = self.transform_embedded_sequences(net)
        net = regression(
            net,
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            name='target')
        model = tflearn.DNN(net, tensorboard_verbose=0)

        if not trainable:
            embedding_matrix = get_embedding_matrix(self.vocab, self.embeddings_path)
            embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
            model.set_weights(embeddingWeights, embedding_matrix)

        return model


    def fit(self, X, y, validation_data=None):

        self.validate_params()

        tf.reset_default_graph()
        model = self.build_model()

        padded_X = pad_sequences(X, self.max_seq_len, padding='pre', truncating='pre')
        one_hot_y = to_categorical(y, nb_classes=self.class_count)

        if validation_data is not None:
            v_X, v_y = validation_data
            v_X = pad_sequences(v_X, self.max_seq_len, padding='pre', truncating='pre')
            v_y = to_categorical(v_y, nb_classes=self.class_count)

            early_stopping = EarlyStoppingCallback(patience=self.patience)
            self.history = model.fit(
                padded_X, one_hot_y,
                n_epoch=self.epochs,
                batch_size=self.batch_size,
                validation_set=(v_X, v_y),
                show_metric=True,
                callbacks=early_stopping)

        else:
            self.history = model.fit(
                padded_X,one_hot_y,
                n_epoch=self.epochs,
                batch_size=self.batch_size,
                show_metric=True)

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
