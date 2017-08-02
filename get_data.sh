cd data

if [ ! -f r8-all-terms.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-train-all-terms.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-test-all-terms.txt
    cat r8-*-all-terms.txt > r8-all-terms.txt
    rm r8-*-all-terms.txt
fi

if [ ! -f r8-no-short.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-train-no-short.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-test-no-short.txt
    cat r8-*-no-short.txt > r8-no-short.txt
    rm r8-*-no-short.txt
fi

if [ ! -f r8-no-stop.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-train-no-stop.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-test-no-stop.txt
    cat r8-*-no-stop.txt > r8-no-stop.txt
    rm r8-*-no-stop.txt
fi

if [ ! -f r8-stemmed.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-train-stemmed.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets//r8-test-stemmed.txt
    cat r8-*-stemmed.txt > r8-stemmed.txt
    rm r8-*-stemmed.txt
fi

if [ ! -f r52-all-terms.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-train-all-terms.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-test-all-terms.txt
    cat r52-*-all-terms.txt > r52-all-terms.txt
    rm r52-*-all-terms.txt
fi

if [ ! -f r52-no-short.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-train-no-short.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-test-no-short.txt
    cat r52-*-no-short.txt > r52-no-short.txt
    rm r52-*-no-short.txt
fi

if [ ! -f r52-no-stop.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-train-no-stop.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-test-no-stop.txt
    cat r52-*-no-stop.txt > r52-no-stop.txt
    rm r52-*-no-stop.txt
fi

if [ ! -f r52-stemmed.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-train-stemmed.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/r52-test-stemmed.txt
    cat r52-*-stemmed.txt > r52-stemmed.txt
    rm r52-*-stemmed.txt
fi

if [ ! -f 20ng-all-terms.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-train-all-terms.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-test-all-terms.txt
    cat 20ng-*-all-terms.txt > 20ng-all-terms.txt
    rm 20ng-*-all-terms.txt
fi

if [ ! -f 20ng-no-short.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-train-no-short.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-test-no-short.txt
    cat 20ng-*-no-short.txt > 20ng-no-short.txt
    rm 20ng-*-no-short.txt
fi

if [ ! -f 20ng-no-stop.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-train-no-stop.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-test-no-stop.txt
    cat 20ng-*-no-stop.txt > 20ng-no-stop.txt
    rm 20ng-*-no-stop.txt
fi

if [ ! -f 20ng-stemmed.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-train-stemmed.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/20ng-test-stemmed.txt
    cat 20ng-*-stemmed.txt > 20ng-stemmed.txt
    rm 20ng-*-stemmed.txt
fi


if [ ! -f webkb-stemmed.txt ]; then
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/webkb-train-stemmed.txt
    wget http://www.cs.umb.edu/~smimarog/textmining/datasets/webkb-test-stemmed.txt
    cat webkb-*-stemmed.txt > webkb-stemmed.txt
    rm webkb-*-stemmed.txt
fi


if [ ! -d glove.6B ]; then
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip
    mkdir glove.6B
    mv glove.6B.* glove.6B
fi


if [ ! -d SentenceCorpus ]; then
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00311/SentenceCorpus.zip
    unzip SentenceCorpus.zip
    rm SentenceCorpus.zip
    rm -rf __MACOSX
fi


if [ ! -d subjectivity_dataset ]; then
    mkdir subjectivity_dataset
    cd subjectivity_dataset
    wget http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz
    tar -xvf rotten_imdb.tar.gz
    cd ../
fi


if [ ! -d polarity ]; then
    mkdir polarity
    cd polarity
    wget http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
    tar -xvf rt-polaritydata.tar.gz
    mv rt-polaritydata/* ./
    rm -rf rt-polaritydata
    cd ../
fi


if [ ! -d stanfordSentimentTreebank ]; then
    wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
    unzip stanfordSentimentTreebank.zip
fi

echo done
