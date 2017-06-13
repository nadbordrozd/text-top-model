import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import Memory

cache = Memory('cache').cache


def read_dataset(path):
    X, y = [], []
    with open(path, "rb") as infile:
        for line in infile:
            label, text = line.split("\t")
            text = text.strip()
            if len(text) == 0:
                continue
            # texts are already tokenized, just split on space
            # in a real case we would use e.g. spaCy for tokenization
            # and maybe remove stopwords etc.
            X.append(text.split())
            y.append(label)
    X, y = np.array(X), np.array(y)
    print "total examples %s" % len(y)
    return X, y


@cache
def prepare_dataset(path):
    X, y = read_dataset(path)
    label_encoder = LabelEncoder()
    token_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(y)
    vocab = list({w for text in X for w in text})
    token_encoder.fit(vocab)
    X = np.array([list(token_encoder.transform(tokens)) for tokens in X])
    return X, labels, token_encoder, label_encoder
