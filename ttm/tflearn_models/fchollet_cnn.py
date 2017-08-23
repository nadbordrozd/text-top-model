from tflearn.layers.core import dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d

from tflearn_text_classifier import TFlearnTextClassifier


class FCholletCNN(TFlearnTextClassifier):
    """Based on
    https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    except with trainable embeddings instead of pretrained from GloVe"""

    def __init__(
            self,
            max_seq_len=1000,
            embedding_dim=30,
            embeddings_path=None,
            optimizer='adam',
            batch_size=32,
            epochs=10,
            filters=128,
            dropout_rate=0,
            vocab_size=None,
            vocab=None,
            class_count=None):
        super(FCholletCNN, self).__init__(
            max_seq_len,
            embedding_dim,
            embeddings_path,
            optimizer,
            batch_size,
            epochs,
            vocab,
            vocab_size,
            class_count)

        self.filters = filters
        self.dropout_rate = 1-dropout_rate
        self.params['filters'] = filters
        self.params['dropout_rate'] = 1-dropout_rate

    def transform_embedded_sequences(self, embedded_sequences):
        net = conv_1d(embedded_sequences, self.filters, 5, 1, activation='relu', padding="valid")
        net = max_pool_1d(net, 5, padding="valid")
        if self.dropout_rate > 0:
            net = dropout(net, self.dropout_rate)
        net = conv_1d(net, self.filters, 5, activation='relu', padding="valid")
        net = max_pool_1d(net, 5, padding="valid")
        if self.dropout_rate > 0:
            net = dropout(net, self.dropout_rate)
        net = conv_1d(net, self.filters, 5, activation='relu', padding="valid")
        net = max_pool_1d(net, 35)
        if self.dropout_rate > 0:
            net = dropout(net, self.dropout_rate)
        net = fully_connected(net, self.filters, activation='relu')
        preds = fully_connected(net, self.class_count, activation='softmax')
        return preds