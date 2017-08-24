import tensorflow as tf
from tflearn.layers import dropout, fully_connected, flatten
from tflearn.layers.conv import conv_1d, max_pool_1d, global_max_pool
from tflearn.layers.merge_ops import merge

from tflearn_text_classifier import TFlearnTextClassifier


class YKimCNN(TFlearnTextClassifier):
    """Based on Alexander Rakhlin's implementation
    https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras
    of Yoon Kim's architecture from the paper
    https://arxiv.org/pdf/1408.5882v2.pdf
    See also:
    https://github.com/tflearn/tflearn/blob/master/examples/nlp/cnn_sentence_classification.py
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
        # remember that dropout:
        # keras  : fraction of the input units to drop.
        # tflearn: probability that each element is kept
        self.params['dropout_rates'] = tuple(1-d for d in dropout_rates)
        self.params['filter_sizes'] = filter_sizes
        self.params['num_filters'] = num_filters
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout_rates = tuple(1-d for d in dropout_rates)
        self.units = units


    def transform_embedded_sequences(self, embedded_sequences):
        drop_1, drop_2 = self.dropout_rates
        net = dropout(embedded_sequences, drop_1)

        conv_blocks = []
        for sz in self.filter_sizes:
            conv = conv_1d(net,
                nb_filter=self.num_filters,
                filter_size=sz,
                padding="valid",
                activation="relu",
                regularizer="L2"
                )
            conv_blocks.append(conv)

        net = merge(conv_blocks, mode='concat', axis=1) if len(conv_blocks) > 1 else conv_blocks[0]
        net = tf.expand_dims(net, 2)
        net = global_max_pool(net)
        net = dropout(net, drop_2)

        model_output = fully_connected(net, self.class_count, activation="softmax")

        return model_output
