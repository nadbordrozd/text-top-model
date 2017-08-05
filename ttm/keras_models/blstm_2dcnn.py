from keras.layers import Dense, LSTM, Bidirectional, Conv2D, Dropout, MaxPool2D, \
    Reshape, Flatten
from keras_text_classifier import KerasTextClassifier


class BLSTM2DCNN(KerasTextClassifier):
    """based on https://arxiv.org/abs/1611.06639v1"""
    def __init__(
            self,
            bidirectional=False,
            max_seq_len=1000,
            embedding_dim=30,
            embeddings_path=None,
            optimizer='adam',
            batch_size=128,
            epochs=10,
            units=128,
            dropout_rate=0.2,
            rec_dropout_rate=0.2,
            conv_filters=32,
            vocab_size=None,
            vocab=None,
            class_count=None
    ):
        super(BLSTM2DCNN, self).__init__(
            max_seq_len,
            embedding_dim,
            embeddings_path,
            optimizer,
            batch_size,
            epochs,
            vocab,
            vocab_size,
            class_count)

        self.params['units'] = units
        self.params['dropout_rate'] = dropout_rate
        self.params['rec_dropout_rate'] = rec_dropout_rate
        self.params['bidirectional'] = bidirectional
        self.params['conv_filters'] = conv_filters

    def transform_embedded_sequences(self, embedded_sequences):
        x = Dropout(self.params['dropout_rate'])(embedded_sequences)
        x = Bidirectional(LSTM(
            self.params['units'],
            dropout=self.params['dropout_rate'],
            recurrent_dropout=self.params['rec_dropout_rate'],
            return_sequences=True))(x)
        x = Reshape((2 * self.max_seq_len, self.params['units'], 1))(x)
        x = Conv2D(self.params['conv_filters'], (3, 3))(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        preds = Dense(self.class_count, activation='softmax')(x)
        return preds
