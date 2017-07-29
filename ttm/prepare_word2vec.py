from gensim.models import Word2Vec

datasets = [
    '20ng-all-terms',
    '20ng-no-short',
    '20ng-no-stop',
    '20ng-stemmed',
    'r52-all-terms',
    'r52-no-short',
    'r52-no-stop',
    'r52-stemmed',
    'r8-all-terms',
    'r8-no-short',
    'r8-no-stop',
    'r8-stemmed',
    'r8-test-all-terms',
    'r8-train-all-terms',
    'webkb-stemmed'
]

dimensionalities = [10, 20, 50, 100]


def make_word2vec(path, dim):
    with open(path, 'rb') as f:
        sentences = [line.split('\t')[1].split() for line in f]

    model = Word2Vec(sentences, size=dim, window=5, min_count=5, workers=4)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    return w2v


def save_word2vec(w2v, out_path):
    with open(out_path, 'wb') as f:
        for word, vec in w2v.items():
            f.write(word + " ")
            f.write(" ".join("%.5f" % x for x in vec))
            f.write('\n')

def main():
    for dataset in datasets:
        for dim in dimensionalities:
            print 'doing', dataset, dim
            path = '../data/%s.txt' % dataset
            out_path = '../data/word2vec/%s_%sd.txt' % (dataset, dim)
            w2v = make_word2vec(path, dim)
            save_word2vec(w2v, out_path)

if __name__ == '__main__':
    main()
