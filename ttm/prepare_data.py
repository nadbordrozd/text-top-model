import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter
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
    labels = label_encoder.fit_transform(y)
    word_counts = Counter(w for text in X for w in text)
    vocab = [w for (w, _) in sorted(word_counts.items(), key=lambda (_, c): -c)]
    word2ind = {w: i for i, w in enumerate(vocab)}
    X = np.array([[word2ind[w] for w in tokens] for tokens in X])
    return X, labels, vocab, label_encoder
