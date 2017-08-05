from keras.layers import Dense, LSTM, Bidirectional

from keras_text_classifier import KerasTextClassifier


class LSTMClassifier(KerasTextClassifier):

    def __init__(
            self,
            bidirectional=False,
            max_seq_len=1000,
            embedding_dim=30,
            embeddings_path=None,
            optimizer='adam',
            batch_size=128,
            epochs=10,
            layers=1,
            units=128,
            dropout_rate=0.2,
            rec_dropout_rate=0.2,
            vocab=None,
            vocab_size=None,
            class_count=None
    ):
        super(LSTMClassifier, self).__init__(
            max_seq_len,
            embedding_dim,
            embeddings_path,
            optimizer,
            batch_size,
            epochs,
            vocab,
            vocab_size,
            class_count)

        self.params['layers'] = layers
        self.params['units'] = units
        self.params['dropout_rate'] = dropout_rate
        self.params['rec_dropout_rate'] = rec_dropout_rate
        self.params['bidirectional'] = bidirectional

    def transform_embedded_sequences(self, embedded_sequences):
        x = embedded_sequences
        for i in range(1, self.params['layers'] + 1):
            # if there are more lstms downstream - return sequences
            return_sequences = i < self.params['layers']
            layer = LSTM(
                self.params['units'],
                dropout=self.params['dropout_rate'],
                recurrent_dropout=self.params['rec_dropout_rate'],
                return_sequences=return_sequences)
            if self.params['bidirectional']:
                x = Bidirectional(layer)(x)
            else:
                x = layer(x)
        preds = Dense(self.class_count, activation='softmax')(x)
        return preds
