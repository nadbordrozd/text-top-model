from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Concatenate

from keras_text_classifier import KerasTextClassifier


class YKimCNN(KerasTextClassifier):
    """Based on Alexander Rakhlin's implementation
    https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras
    of Yoon Kim's architecture from the paper
    https://arxiv.org/pdf/1408.5882v2.pdf
    """

    def __init__(
            self,
            max_seq_len=50,
            embedding_dim=30,
            embeddings_path=None,
            optimizer='adam',
            batch_size=32,
            epochs=10,
            units=64,
            dropout_rates=(0.5, 0.8),
            filter_sizes=(3, 8),
            num_filters=10,
            vocab_size=None,
            vocab=None,
            class_count=None):
        super(YKimCNN, self).__init__(
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
        self.params['dropout_rates'] = dropout_rates
        self.params['filter_sizes'] = filter_sizes
        self.params['num_filters'] = num_filters
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout_rates = dropout_rates
        self.units = units

    def transform_embedded_sequences(self, embedded_sequences):
        drop_1, drop_2 = self.dropout_rates
        z = Dropout(drop_1)(embedded_sequences)

        conv_blocks = []
        for sz in self.filter_sizes:
            conv = Conv1D(
                filters=self.num_filters,
                kernel_size=sz,
                padding="valid",
                activation="relu",
                strides=1)(z)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        z = Dropout(drop_2)(z)
        z = Dense(self.units, activation="relu")(z)
        model_output = Dense(self.class_count, activation="softmax")(z)
        return model_output
