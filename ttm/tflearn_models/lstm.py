from tflearn.layers.recurrent import lstm
from tflearn.layers.core import fully_connected


from tflearn_text_classifier import TFlearnTextClassifier


class LSTMClassifier(TFlearnTextClassifier):

    def __init__(
            self,
            # we will see in the future how we code the bidirectional rnn with tflearn
            # bidirectional=False,
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
        # with tflearn dropout is the probability that each element is kept,
        # while in keras is the fraction of the input units to drop, so:
        self.params['dropout_rate'] = 1-dropout_rate
        self.params['rec_dropout_rate'] = 1-rec_dropout_rate
        # self.params['bidirectional'] = bidirectional

    def transform_embedded_sequences(self, embedded_sequences):
        net = embedded_sequences
        for i in range(1, self.params['layers'] + 1):
            # if there are more lstms downstream - return sequences
            return_sequences = i < self.params['layers']
            net = lstm(net,
                self.params['units'],
                dropout=(self.params['dropout_rate'],self.params['rec_dropout_rate']),
                return_seq=return_sequences)

        preds = fully_connected(net, self.class_count, activation='softmax')
        return preds
