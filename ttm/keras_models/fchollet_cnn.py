from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D

from keras_text_classifier import KerasTextClassifier


class FCholletCNN(KerasTextClassifier):
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
            units=128,
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

        self.units = units
        self.dropout_rate = dropout_rate
        self.params['units'] = units
        self.params['dropout_rate'] = dropout_rate

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
        preds = Dense(self.class_count, activation='softmax')(x)
        return preds
