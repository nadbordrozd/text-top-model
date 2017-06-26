from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D

from embedding_classifier import EmbeddingClassifier


class FCholletCNN(EmbeddingClassifier):
    """Based on
    https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    except with trainable embeddings instead of pretrained from GloVe"""

    def __init__(self, epochs=20, embedding_dim=20, embeddings_path=None, max_seq_len=1000,
                 optimizer='adam',
                 units=128,
                 dropout_rate=0):
        super(FCholletCNN, self).__init__(epochs, embedding_dim, embeddings_path, max_seq_len, optimizer)

        self.units = units
        self.dropout_rate = dropout_rate

    def transform_embedded_sequences(self, embedded_sequences):
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
        return preds
